"""
Train UNI2-h only baseline with 3 MLP layers
"""

import sys
sys.path.append("/orcd/data/edboyden/002/ezh/deeplearning/RandStainNA")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from randstainna import RandStainNA
import timm
import timm.layers
import os
import json
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


# preprocessing

class RandStainNATransform(A.ImageOnlyTransform):
    """Wrapper for RandStainNA to work with albumentations"""
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


# instantiate RandStainNA transform
randstain_transform = RandStainNATransform(
    yaml_file="/orcd/data/edboyden/002/ezh/deeplearning/RandStainNA/CRC_LAB_randomTrue_n0.yaml",
    std_hyper=-0.3,
    probability=1.0,
    distribution="normal",
)

# training transforms (with augmentation + ImageNet normalization for UNI)
train_transform = A.Compose([
    A.RandomRotate90(p=0.75),
    A.Flip(p=0.5),
    A.GaussianBlur(blur_limit=(1, 3), p=0.3),
    randstain_transform,
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# validation transforms (no augmentation)
val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


class MIDOGDataset(Dataset):
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
            img = self.transform(image=img)['image']
        
        return img, label




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


class UNI2hClassifier(nn.Module):
    """
    UNI2-h feature extractor + classifier.
    Classifier head matches hybrid model for fair comparison.
    """
    def __init__(self, num_classes=2, uni_dim=1536):
        super().__init__()
        
        self.uni = load_uni2h()
        
        # freeze UNI2-h
        for param in self.uni.parameters():
            param.requires_grad = False
        self.uni.eval()
        
        # classifier head 
        self.classifier = nn.Sequential(
            nn.Linear(uni_dim, 512),
            nn.GELU(),  
            nn.Dropout(0.3),
            nn.Linear(512, 256),  
            nn.GELU(),            
            nn.Dropout(0.3),      
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        with torch.no_grad():
            features = self.uni(x)  # [batch, 1536]
        
        out = self.classifier(features)
        return out
    
    def train(self, mode=True):
        """Override to keep UNI2-h frozen in eval mode"""
        super().train(mode)
        self.uni.eval()
        return self




def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for imgs, labels in tqdm(loader, desc="Training"):
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
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
        for imgs, labels in tqdm(loader, desc="Validation"):
            imgs, labels = imgs.to(device), labels.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    return total_loss / len(loader), all_preds, all_labels, all_probs




def main():
    config = {
        'batch_size': 32,
        'num_epochs': 15,  
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'uni_dim': 1536,
        'num_classes': 2,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # create datasets
    train_dataset = MIDOGDataset(
        '/orcd/data/edboyden/002/ezh/deeplearning/training/data/train', 
        transform=train_transform
    )
    val_dataset = MIDOGDataset(
        '/orcd/data/edboyden/002/ezh/deeplearning/training/data/val', 
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=4
    )
    
    # create model
    model = UNI2hClassifier(
        num_classes=config['num_classes'],
        uni_dim=config['uni_dim']
    ).to(device)
    
    # loss and optimizer 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
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
        
        # train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # validate
        val_loss, val_preds, val_labels, val_probs = validate(model, val_loader, criterion, device)
        
        # compute metrics
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
        
        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_roc_auc': val_roc_auc,
                'config': config,
            }, 'model_uni2h_3mlp_best.pth')
            print(f"Saved best model (val_acc: {val_acc:.4f})")
    
    # save training history
    with open('uni2h_3mlp_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("\nSaved training history to uni2h_3mlp_training_history.json")
    

    print("Training complete")
    print(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch+1})")


if __name__ == '__main__':
    main()