"""
Classifier for Hybrid Model: UNI2-h + CNN with Cross-Branch Attention
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

from cnn_cba import CNNBranch, CrossBranchAttention
from hybrid_dataloader import get_dataloaders


def load_uni2h():
    """
    Load UNI2-h
    """
    from huggingface_hub import hf_hub_download
    
    # UNI2-h specific architecture parameters
    timm_kwargs = {
        'img_size': 224,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667 * 2,  # ~5.333
        'num_classes': 0,  # feature extractor only
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked,
        'act_layer': torch.nn.SiLU,
        'reg_tokens': 8,
        'dynamic_img_size': True,
    }
    
    # direct hf-hub loading with full kwargs
    model = timm.create_model(
        "hf-hub:MahmoodLab/UNI2-h",
        pretrained=True,
        **timm_kwargs
    )
    print("loaded UNI2-h from hub")
    return model
    

class HybridClassifier(nn.Module):
    """
    UNI2-h + CNN with Cross-Branch Attention
    """
    def __init__(
        self,
        num_classes: int = 2,
        uni_dim: int = 1536,           # UNI2-h embedding dimension
        cnn_out_channels: int = 256,   # CNN output channels (dim_kv for CBA)
        cba_hidden_dim: int = 256,     # Per-head hidden dimension in CBA
        cba_num_heads: int = 4,        # Number of attention heads in CBA
    ):
        super().__init__()
        
        # uni branch (frozen)
        self.uni = load_uni2h()
        
        # freeze uni2-h weights
        for param in self.uni.parameters():
            param.requires_grad = False
        self.uni.eval()  # keep in eval mode
        
        self.uni_dim = uni_dim
        
        # cnn branch
        self.cnn_branch = CNNBranch(
            in_channels=3,
            base_channels=32,
            out_channels=cnn_out_channels
        )
        
        # cross-branch attention
        self.cba = CrossBranchAttention(
            dim_q=uni_dim,
            dim_kv=cnn_out_channels,
            n_hidden=cba_hidden_dim,
            num_heads=cba_num_heads
        )
        
        # feature fusion
        # Combine CBA-refined tokens with CNN global features
        # CBA output: (B, T, uni_dim) -> global pool -> (B, uni_dim)
        # CNN output: (B, C, 16, 16) -> global pool -> (B, C)
        fused_dim = uni_dim + cnn_out_channels
        
        # classification head
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
            img_uni: (B, 3, 224, 224) - normalized image for UNI2-h
            img_cnn: (B, 3, 224, 224) - raw image (scaled to [0,1]) for CNN
        
        Returns:
            logits: (B, num_classes)
            attn_weights: (B, num_heads, T, HW) - attention maps for visualization
        """
        batch_size = img_uni.shape[0]
        
        # uni2-h forward (frozen)
        with torch.no_grad():
            # get intermediate features (all tokens including register tokens)
            uni_features = self.uni.forward_features(img_uni)  # (B, T, 1536)
            # T = 256 patches + 8 register tokens = 264 for UNI2-h
        
        # cnn forward (learnable)
        cnn_features = self.cnn_branch(img_cnn)  # (B, C, 16, 16)
        
        # cross-branch attention
        # UNI tokens query CNN features for local detail
        refined_tokens, attn_weights = self.cba(uni_features, cnn_features)  # (B, T, 1536)
        
        # feature pooling
        # global average pool over refined tokens
        uni_pooled = refined_tokens.mean(dim=1)  # (B, 1536)
        
        # global average pool over cnn spatial features
        cnn_pooled = F.adaptive_avg_pool2d(cnn_features, 1).squeeze(-1).squeeze(-1)  # (B, C)
        
        # fusion & classification
        fused = torch.cat([uni_pooled, cnn_pooled], dim=1)  # (B, 1536 + C)
        logits = self.classifier(fused)  # (B, num_classes)
        
        return logits, attn_weights
    
    def train(self, mode=True):
        """Override to keep UNI in eval mode"""
        super().train(mode)
        self.uni.eval()  
        return self


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
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
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    """Validate the model, returning predictions for metric computation"""
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
            all_probs.extend(probs[:, 1].cpu().numpy())  # prob of positive class
    
    return total_loss / len(loader), all_preds, all_labels, all_probs


def main():
    config = {
        'train_dir': '/orcd/data/edboyden/002/ezh/deeplearning/training/data/train',
        'val_dir': '/orcd/data/edboyden/002/ezh/deeplearning/training/data/val',
        'batch_size': 32,
        'num_epochs': 15,
        'learning_rate': 1e-4,
        'num_workers': 4,
        
        # model hyperparameters
        'uni_dim': 1536,           # UNI2-h embedding dimension
        'cnn_out_channels': 256,   # CNN output channels
        'cba_hidden_dim': 256,     # CBA per-head hidden dim
        'cba_num_heads': 4,        # number of CBA heads
        'num_classes': 2,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # data loading
    print("\nLoading datasets")
    train_loader, val_loader = get_dataloaders(
        config['train_dir'],
        config['val_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )
    
    # model initialization
    print("\nInitializing model")
    model = HybridClassifier(
        num_classes=config['num_classes'],
        uni_dim=config['uni_dim'],
        cnn_out_channels=config['cnn_out_channels'],
        cba_hidden_dim=config['cba_hidden_dim'],
        cba_num_heads=config['cba_num_heads'],
    ).to(device)
    
    # count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # loss & optimizer
    criterion = nn.CrossEntropyLoss()
    
    # only optimize trainable parameters (cnn + cba + classifier)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config['learning_rate'],
        weight_decay=0.01
    )
    
    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=1e-6
    )
    
    # training loop
    print("\nStarting training")
    best_val_acc = 0
    
    # metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_roc_auc': [],
        'lr': [],
        'epoch_time': [],
    }
    
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning rate: {current_lr:.2e}")
        
        # train epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # validate epoch (now returns predictions)
        val_loss, val_preds, val_labels, val_probs = validate(model, val_loader, criterion, device)
        
        # compute metrics
        val_acc = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, zero_division=0)
        val_recall = recall_score(val_labels, val_preds, zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        val_roc_auc = roc_auc_score(val_labels, val_probs)
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        # store metrics
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
        print(f"Epoch time: {epoch_time:.1f}s")
        
        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_roc_auc': val_roc_auc,
                'config': config,
            }, 'hybrid_best.pth')
    
    # save training history
    with open('hybrid_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("\nSaved training history to hybrid_training_history.json")
    

    print("Training complete")

if __name__ == '__main__':
    main()