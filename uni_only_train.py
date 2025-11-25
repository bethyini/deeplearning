"""
Train UNI
Includes all preprocessing and augmentation
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from randstainna import RandStainNA
import timm
import os

# preprocessing

# instantiate RandStainNA 
randstain = RandStainNA(
    yaml_file="/orcd/data/edboyden/002/ezh/deeplearning/RandStainNA/CRC_LAB_randomTrue_n0.yaml",
    std_hyper=-0.3,
    probability=1.0,
    distribution="normal",
    is_train=True,
)

# Training transforms (with augmentation)
train_transform = A.Compose([
    A.RandomRotate90(p=0.75),
    A.Flip(p=0.5),
    A.GaussianBlur(blur_limit=(1, 3), p=0.3),
    A.Lambda(image=lambda img, **kwargs: randstain(img)),
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Validation transforms (no augmentation)
val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


class MIDOGDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        data_dir: 'data/train' or 'data/val'
        """
        self.transform = transform
        
        # Load all patches
        mitosis_dir = Path(data_dir) / 'mitosis'
        hard_neg_dir = Path(data_dir) / 'hard_negative'
        
        self.paths = []
        self.labels = []
        
        # Mitosis patches (label=1)
        for p in mitosis_dir.glob('*.png'):
            self.paths.append(str(p))
            self.labels.append(1)
        
        # Hard negative patches (label=0)
        for p in hard_neg_dir.glob('*.png'):
            self.paths.append(str(p))
            self.labels.append(0)
        
        print(f"Loaded {len(self.paths)} patches")
        print(f"  Mitosis: {sum(self.labels)}")
        print(f"  Hard negatives: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        # Load image
        img = np.array(Image.open(self.paths[idx]).convert('RGB'))
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            img = self.transform(image=img)['image']
        
        return img, label


class UNIClassifier(nn.Module):
    """UNI feature extractor + classifier"""
    def __init__(self, uni_path='./uni_weights/pytorch_model.bin', num_classes=2):
        super().__init__()
        
        # Load UNI
        self.uni = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True
        )
        
        # Load pretrained weights
        state_dict = torch.load(uni_path, map_location='cpu')
        self.uni.load_state_dict(state_dict)
        
        # Freeze UNI
        for param in self.uni.parameters():
            param.requires_grad = False
        
        # Classifier head (trainable)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Extract UNI features (frozen)
        with torch.no_grad():
            features = self.uni(x)  # [batch, 1024]
        
        # Classify (trainable)
        out = self.classifier(features)
        return out

# training

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
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validation"):
            imgs, labels = imgs.to(device), labels.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(loader), correct / total


def main():
    # Config
    batch_size = 32
    num_epochs = 10
    lr = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = MIDOGDataset('data/train', transform=train_transform)
    val_dataset = MIDOGDataset('data/val', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4)
    
    # Create model
    print("\nInitializing model")
    model = UNIClassifier(uni_path='/orcd/data/edboyden/002/ezh/deeplearning/training/uni_weights/pytorch_model.bin').to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    
    # Training loop
    print("\nStarting training")
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'model_a_uni_best.pth')
            print(f"Saved best model (val_acc: {val_acc:.4f})")
    
    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")

if __name__ == '__main__':
    main()