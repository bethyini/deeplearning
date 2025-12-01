"""
CNN-only control
Fixed hyperparameters for training from scratch
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
import json
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


class ConvBlock(nn.Module):
    # in_ch = num input channels
    # out_ch = num output channels
    # kernel_size = size of conv filter
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)  # 2D conv layer
        self.bn = nn.BatchNorm2d(out_ch)  # batch normalization layer
        self.act = nn.GELU()  # GELU activation like SAMUS

    def forward(self, x):
        # conv -> batch normalization -> activation
        return self.act(self.bn(self.conv(x)))


# Conv + Pool block
class ConvPoolBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # max pooling layer, reduces spatial dims by half each time

    def forward(self, x):
        # conv -> max pooling
        return self.pool(self.conv(x))


class CNNClassifier(nn.Module):
    """
    CNN-only classifier using the same architecture as hybrid model's CNN branch.
    """
    def __init__(self, in_channels=3, base_channels=64, feature_dim=512, num_classes=2):
        super().__init__()
        
        # Encoder path (progressively downsample spatial dims)
        self.layer1 = ConvBlock(in_channels, base_channels)        # 224 -> 224
        self.layer2 = ConvPoolBlock(base_channels, base_channels)  # 224 -> 112
        self.layer3 = ConvPoolBlock(base_channels, base_channels * 2)   # 112 -> 56
        self.layer4 = ConvPoolBlock(base_channels * 2, base_channels * 4)  # 56 -> 28
        self.layer5 = ConvPoolBlock(base_channels * 4, base_channels * 4)  # 28 -> 14
        
        # Feature refinement with residual connections (two conv blocks, doesn't change spatial dims)
        self.refine = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 4),
            ConvBlock(base_channels * 4, base_channels * 4)
        )
        
        # project to feature dimension
        self.proj = nn.Conv2d(base_channels * 4, feature_dim, kernel_size=1)
        
        # Global average pooling + classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights for training from scratch
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training from scratch"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder
        x = self.layer1(x)  # B×3×224×224 → B×64×224×224
        x = self.layer2(x)  # B×64×224×224 → B×64×112×112
        x = self.layer3(x)  # B×64×112×112 → B×128×56×56
        x = self.layer4(x)  # B×128×56×56 → B×256×28×28
        x = self.layer5(x)  # B×256×28×28 → B×256×14×14
        
        # Residual refinement
        identity = x
        x = self.refine(x) + identity
        
        # Project
        x = self.proj(x)  # B×feature_dim×14×14
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, 1)  # B×feature_dim×1×1
        x = x.view(x.size(0), -1)  # B×feature_dim
        
        # Classify
        logits = self.classifier(x)  # B×num_classes
        
        return logits


# preprocessing for CNN

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

# training transforms - use ImageNet normalization (works better for CNNs from scratch)
train_transform = A.Compose([
    A.RandomRotate90(p=0.75),
    A.Flip(p=0.5),
    A.GaussianBlur(blur_limit=(1, 3), p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),  # Added augmentation
    randstain_transform,
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406),   # ImageNet normalization
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
        
        # mitosis patches (label=1)
        for p in mitosis_dir.glob('*.png'):
            self.paths.append(str(p))
            self.labels.append(1)
        
        # hard negative patches (label=0)
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


# training and validation functions
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
        
        # gradient clipping for stability
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
        'num_epochs': 30,       
        'learning_rate': 1e-3,  
        'weight_decay': 1e-4,   
        'base_channels': 64,    
        'feature_dim': 512,     
        'num_classes': 2,
        'warmup_epochs': 3,     
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\nConfig: {config}")
    
    # Create datasets
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
    
    # Create model
    model = CNNClassifier(
        in_channels=3,
        base_channels=config['base_channels'],
        feature_dim=config['feature_dim'],
        num_classes=config['num_classes']
    ).to(device)
    
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < config['warmup_epochs']:
            # Linear warmup
            return (epoch + 1) / config['warmup_epochs']
        else:
            # Cosine decay
            progress = (epoch - config['warmup_epochs']) / (config['num_epochs'] - config['warmup_epochs'])
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    

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
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_preds, val_labels, val_probs = validate(model, val_loader, criterion, device)
        
        # Compute metrics
        val_acc = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, zero_division=0)
        val_recall = recall_score(val_labels, val_preds, zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        val_roc_auc = roc_auc_score(val_labels, val_probs)
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        # Store history
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
        print(f"Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        print(f"Val ROC-AUC: {val_roc_auc:.4f}")
        print(f"Epoch time: {epoch_time:.1f}s")
        
        # Save best model
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
            }, 'model_cnn_only_best.pth')
            print(f"Saved best model (val_f1: {val_f1:.4f})")
    

    with open('cnn_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("\nSaved training history to cnn_training_history.json")
    
    print("Training complete")
    print(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch+1})")


if __name__ == '__main__':
    main()