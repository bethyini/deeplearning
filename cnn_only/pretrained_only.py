"""
pretrained CNN-only baseline using backbone EfficientNet-B3 pretrained on ImageNet
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
import json
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


class PretrainedCNNClassifier(nn.Module):
    """
    CNN classifier using pretrained ImageNet backbone.
    Based on MIDOG challenge winning approaches that use:
    - EfficientNet-B3/B5 
    - ResNet-50/101
    """
    def __init__(self, 
                 backbone='efficientnet_b3',  
                 num_classes=2, 
                 pretrained=True,
                 freeze_backbone=False):
        super().__init__()
        
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # remove classifier, get features only
        )
        
        feature_dim = self.backbone.num_features
        
        # classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # extract features
        features = self.backbone(x)  
        
        # classify
        logits = self.classifier(features)
        
        return logits


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


randstain_transform = RandStainNATransform(
    yaml_file="/orcd/data/edboyden/002/ezh/deeplearning/RandStainNA/CRC_LAB_randomTrue_n0.yaml",
    std_hyper=-0.3,
    probability=1.0,
    distribution="normal",
)

# training transforms with ImageNet normalization
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

# validation transforms
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
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        'backbone': 'efficientnet_b3',  
        'pretrained': True,
        'freeze_backbone': False,  # fine-tune model
        'num_classes': 2,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\nConfig: {config}")
    
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
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # create model
    model = PretrainedCNNClassifier(
        backbone=config['backbone'],
        num_classes=config['num_classes'],
        pretrained=config['pretrained'],
        freeze_backbone=config['freeze_backbone']
    ).to(device)
    
    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
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
    best_val_f1 = 0
    
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
        
        current_lr = optimizer.param_groups[0]['lr']
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
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
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
            }, f'model_{config["backbone"]}_best.pth')
            print(f"Saved best model (val_f1: {val_f1:.4f})")
    
    # save training history
    history_file = f'pretrained_cnn_training_history.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved training history to {history_file}")
    
    print("Training complete")
    print(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch+1})")


if __name__ == '__main__':
    main()
