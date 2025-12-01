"""
UNI2-h + Pretrained EfficientNet-B3 CNN with Cross-Branch Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import timm
import timm.layers
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import json
import time
import os


def load_uni2h():    
    timm_kwargs = {
        'img_size': 224,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667 * 2,
        'num_classes': 0,
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked,
        'act_layer': torch.nn.SiLU,
        'reg_tokens': 8,
        'dynamic_img_size': True,
    }
    
    model = timm.create_model(
        "hf-hub:MahmoodLab/UNI2-h",
        pretrained=True,
        **timm_kwargs
    )
    print("Successfully loaded UNI2-h")
    return model


class PretrainedCNNBranch(nn.Module):
    """
    CNN branch using pretrained EfficientNet backbone.
    """
    def __init__(self, backbone='efficientnet_b3', out_channels=256):
        super().__init__()
        
        # load pretrained backbone 
        self.backbone = timm.create_model(
            backbone,
            pretrained=True,
            features_only=True,  # get intermediate feature maps
            out_indices=[4],     # last feature map (before pooling)
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy)
            backbone_channels = features[0].shape[1]
            spatial_size = features[0].shape[2]
        
        self.proj = nn.Sequential(
            nn.Conv2d(backbone_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        
        self.out_channels = out_channels
        self.spatial_size = spatial_size
    
    def forward(self, x):
        # extract features from backbone
        features = self.backbone(x)[0]  # (B, C, H, W)
        
        # project to output channels
        out = self.proj(features)  # (B, out_channels, H, W)
        
        # resize to 16x16 to match UNI2-h patch grid
        if out.shape[2] != 16:
            out = F.interpolate(out, size=(16, 16), mode='bilinear', align_corners=False)
        
        return out


class CrossBranchAttention(nn.Module):
    """
    Cross-branch attention: UNI tokens query CNN features for local detail.
    """
    def __init__(self, dim_q, dim_kv, n_hidden, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = n_hidden // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query from UNI tokens
        self.q_proj = nn.Linear(dim_q, n_hidden)
        
        # Key, Value from CNN features
        self.k_proj = nn.Linear(dim_kv, n_hidden)
        self.v_proj = nn.Linear(dim_kv, n_hidden)
        
        # Output projection back to UNI dimension
        self.out_proj = nn.Linear(n_hidden, dim_q)
        
        # Layer norm and residual
        self.norm = nn.LayerNorm(dim_q)
    
    def forward(self, uni_tokens, cnn_features):
        """
        Args:
            uni_tokens: (B, T, dim_q) - UNI2-h token features
            cnn_features: (B, C, H, W) - CNN spatial features
        Returns:
            refined_tokens: (B, T, dim_q)
            attn_weights: (B, num_heads, T, HW)
        """
        B, T, _ = uni_tokens.shape
        _, C, H, W = cnn_features.shape
        
        # Flatten CNN features: (B, C, H, W) -> (B, HW, C)
        cnn_flat = cnn_features.flatten(2).transpose(1, 2)  # (B, HW, C)
        
        # Project queries, keys, values
        Q = self.q_proj(uni_tokens)  # (B, T, n_hidden)
        K = self.k_proj(cnn_flat)     # (B, HW, n_hidden)
        V = self.v_proj(cnn_flat)     # (B, HW, n_hidden)
        
        # Reshape for multi-head attention
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)    # (B, heads, T, head_dim)
        K = K.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, HW, head_dim)
        V = V.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, HW, head_dim)
        
        # Attention
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B, heads, T, HW)
        attn_weights = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = attn_weights @ V  # (B, heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, n_hidden)
        
        # Project back and residual connection
        out = self.out_proj(out)
        refined_tokens = self.norm(uni_tokens + out)
        
        return refined_tokens, attn_weights


class HybridClassifierV2(nn.Module):
    """
    UNI2-h + Pretrained CNN with Cross-Branch Attention
    """
    def __init__(
        self,
        num_classes: int = 2,
        uni_dim: int = 1536,
        cnn_backbone: str = 'efficientnet_b3',
        cnn_out_channels: int = 256,
        cba_hidden_dim: int = 256,
        cba_num_heads: int = 4,
    ):
        super().__init__()
        
        # UNI2-h branch (frozen)
        self.uni = load_uni2h()
        for param in self.uni.parameters():
            param.requires_grad = False
        self.uni.eval()
        self.uni_dim = uni_dim
        
        # Pretrained CNN branch (trainable)
        self.cnn_branch = PretrainedCNNBranch(
            backbone=cnn_backbone,
            out_channels=cnn_out_channels,
        )
        
        # Cross-branch attention
        self.cba = CrossBranchAttention(
            dim_q=uni_dim,
            dim_kv=cnn_out_channels,
            n_hidden=cba_hidden_dim,
            num_heads=cba_num_heads
        )
        
        # Classification head
        fused_dim = uni_dim + cnn_out_channels
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, img_uni: torch.Tensor, img_cnn: torch.Tensor):
        """
        Args:
            img_uni: (B, 3, 224, 224) - ImageNet-normalized for UNI2-h
            img_cnn: (B, 3, 224, 224) - ImageNet-normalized for CNN
        """
        # UNI2-h forward (frozen)
        with torch.no_grad():
            uni_features = self.uni.forward_features(img_uni)  # (B, T, 1536)
        
        # CNN forward (pretrained, may be trainable)
        cnn_features = self.cnn_branch(img_cnn)  # (B, C, 16, 16)
        
        # Cross-branch attention
        refined_tokens, attn_weights = self.cba(uni_features, cnn_features)
        
        # Feature pooling
        uni_pooled = refined_tokens.mean(dim=1)  # (B, 1536)
        cnn_pooled = F.adaptive_avg_pool2d(cnn_features, 1).squeeze(-1).squeeze(-1)  # (B, C)
        
        # Fusion & classification
        fused = torch.cat([uni_pooled, cnn_pooled], dim=1)
        logits = self.classifier(fused)
        
        return logits, attn_weights
    
    def train(self, mode=True):
        """Override to keep UNI in eval mode"""
        super().train(mode)
        self.uni.eval()
        return self


# data loading

import sys
sys.path.append("/orcd/data/edboyden/002/ezh/deeplearning/RandStainNA")

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from randstainna import RandStainNA

# preprocessing
class RandStainNATransform(A.ImageOnlyTransform):
    def __init__(self, yaml_file, std_hyper=-0.3, probability=1.0, 
                 distribution="normal", always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.randstain = RandStainNA(
            yaml_file=yaml_file,
            std_hyper=std_hyper,
            probability=probability,
            distribution=distribution,
            is_train=True,
        )
    
    def apply(self, img, **params):
        return self.randstain(img)


randstain_transform = RandStainNATransform(
    yaml_file="/orcd/data/edboyden/002/ezh/deeplearning/RandStainNA/CRC_LAB_randomTrue_n0.yaml",
    std_hyper=-0.3,
    probability=1.0,
    distribution="normal",
)

# both branches use ImageNet normalization
train_transform = A.Compose([
    A.RandomRotate90(p=0.75),
    A.Flip(p=0.5),
    A.GaussianBlur(blur_limit=(1, 3), p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    randstain_transform,
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


class MIDOGDataset(Dataset):
    """Single-image dataset (same image for both branches)"""
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        
        mitosis_dir = Path(data_dir) / 'mitosis'
        hard_neg_dir = Path(data_dir) / 'hard_negative'
        
        self.paths = []
        self.labels = []
        
        for p in mitosis_dir.glob('*.png'):
            self.paths.append(str(p))
            self.labels.append(1)
        
        for p in hard_neg_dir.glob('*.png'):
            self.paths.append(str(p))
            self.labels.append(0)
        
        print(f"Loaded {len(self.paths)} patches")
        print(f"  Mitosis: {sum(self.labels)}")
        print(f"  Hard negatives: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = np.array(Image.open(self.paths[idx]).convert('RGB'))
        label = self.labels[idx]
        
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']
        
        # same image for both branches
        return img, img, label




def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for img_uni, img_cnn, labels in tqdm(loader, desc="Training"):
        img_uni = img_uni.to(device)
        img_cnn = img_cnn.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(img_uni, img_cnn)
        loss = criterion(logits, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for img_uni, img_cnn, labels in tqdm(loader, desc="Validation"):
            img_uni = img_uni.to(device)
            img_cnn = img_cnn.to(device)
            labels = labels.to(device)
            
            logits, _ = model(img_uni, img_cnn)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            probs = F.softmax(logits, dim=1)
            _, predicted = logits.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    return total_loss / len(loader), all_preds, all_labels, all_probs



def main():
    config = {
        'train_dir': '/orcd/data/edboyden/002/ezh/deeplearning/training/data/train',
        'val_dir': '/orcd/data/edboyden/002/ezh/deeplearning/training/data/val',
        'batch_size': 32,
        'num_epochs': 15,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        
        # model config
        'uni_dim': 1536,
        'cnn_backbone': 'efficientnet_b3',  
        'cnn_out_channels': 256,
        'cba_hidden_dim': 256,
        'cba_num_heads': 4,
        'num_classes': 2,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\nConfig: {config}")
    
    # data loading
    train_dataset = MIDOGDataset(config['train_dir'], transform=train_transform)
    val_dataset = MIDOGDataset(config['val_dir'], transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=4, pin_memory=True)
    

    model = HybridClassifierV2(
        num_classes=config['num_classes'],
        uni_dim=config['uni_dim'],
        cnn_backbone=config['cnn_backbone'],
        cnn_out_channels=config['cnn_out_channels'],
        cba_hidden_dim=config['cba_hidden_dim'],
        cba_num_heads=config['cba_num_heads'],
    ).to(device)

    
    # optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs'], eta_min=1e-6
    )
    
    # training loop
    print("\nStarting training")
    best_val_f1 = 0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_roc_auc': [],
        'lr': [], 'epoch_time': [],
    }
    
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.2e}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds, val_labels, val_probs = validate(model, val_loader, criterion, device)
        
        # metrics
        val_acc = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, zero_division=0)
        val_recall = recall_score(val_labels, val_preds, zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        val_roc_auc = roc_auc_score(val_labels, val_probs)
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        # store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['val_roc_auc'].append(val_roc_auc)
        history['lr'].append(current_lr)
        history['epoch_time'].append(epoch_time)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val ROC-AUC: {val_roc_auc:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch
            best_conf_matrix = confusion_matrix(val_labels, val_preds)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_roc_auc': val_roc_auc,
                'config': config,
            }, 'hybrid_pretrained_best.pth')
            print(f"Saved best model (val_f1: {val_f1:.4f})")
    
    # save history
    with open('hybrid_pretrained_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("Training complete")
    print(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch+1})")


if __name__ == '__main__':
    main()
