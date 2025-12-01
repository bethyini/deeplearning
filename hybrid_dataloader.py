"""
Data Preprocessing and Loading for Hybrid Model (UNI2-h + CNN)
"""
import sys
sys.path.append("/orcd/data/edboyden/002/ezh/deeplearning/RandStainNA")

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from randstainna import RandStainNA

# preprocessing

# wrapper class for RandStainNA to work with albumentations and multiprocessing
# only used for UNI branch
class RandStainNATransform(A.ImageOnlyTransform):
    """
    Albumentations-compatible wrapper for RandStainNA.
    """
    def __init__(self, yaml_file, std_hyper=-0.3, probability=1.0, 
                 distribution="normal", is_train=True, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.randstain = RandStainNA(
            yaml_file=yaml_file,
            std_hyper=std_hyper,
            probability=probability,
            distribution=distribution,
            is_train=is_train,
        )
    
    def apply(self, img, **params):
        return self.randstain(img)


class MIDOGHybridDataset(Dataset):
    """
    Dataset with synchronized augmentations between UNI and CNN branches.
    
    Uses Albumentations ReplayCompose to ensure the same spatial transforms
    (rotation, flip) are applied to both versions of the image.
    
    Post-processing differs per branch:
    - UNI: Gaussian blur + stain augmentation + ImageNet normalization
    - CNN: raw pixels scaled to [0, 1]
    """
    def __init__(self, data_dir, is_train=True):
        self.is_train = is_train
        
        # shared spatial augmentations (will be replayed for both branches)
        if is_train:
            self.spatial_transform = A.ReplayCompose([
                A.RandomRotate90(p=0.75),
                A.Flip(p=0.5),
            ])
        else:
            self.spatial_transform = A.ReplayCompose([])  # no augmentation for val
        
        # uni-specific transforms (applied after spatial)
        if is_train:
            self.uni_post = A.Compose([
                A.GaussianBlur(blur_limit=(1, 3), p=0.3),
                RandStainNATransform(
                    yaml_file="/orcd/data/edboyden/002/ezh/deeplearning/RandStainNA/CRC_LAB_randomTrue_n0.yaml",
                    std_hyper=-0.3,
                    distribution="normal",
                    is_train=True,
                    p=1.0
                ),
                A.Resize(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406), 
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.uni_post = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406), 
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        
        # cnn-specific transforms (applied after spatial)
        self.cnn_post = A.Compose([
            A.Resize(224, 224),
            A.ToFloat(max_value=255.0),  # Scale to [0, 1]
            ToTensorV2(),
        ])
        
        # load all patches
        mitosis_dir = Path(data_dir) / 'mitosis'
        hard_neg_dir = Path(data_dir) / 'hard_negative'
        
        self.paths = []
        self.labels = []
        
        # mitosis patches (positive class)
        for p in mitosis_dir.glob('*.png'):
            self.paths.append(str(p))
            self.labels.append(1)
        
        # hard negative patches (negative class)
        for p in hard_neg_dir.glob('*.png'):
            self.paths.append(str(p))
            self.labels.append(0)
        
        print(f"Loaded {len(self.paths)} patches")
        print(f"  Mitosis: {sum(self.labels)}")
        print(f"  Hard negatives: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        # load image as numpy array
        img = np.array(Image.open(self.paths[idx]).convert('RGB'))
        label = self.labels[idx]
        
        # apply spatial augmentation and capture the replay
        result = self.spatial_transform(image=img)
        img_spatial = result['image']
        replay = result['replay']
        
        # apply the same spatial augmentation to get a copy for CNN
        img_spatial_cnn = A.ReplayCompose.replay(replay, image=img)['image']
        
        # apply branch-specific post-processing
        img_uni = self.uni_post(image=img_spatial)['image']
        img_cnn = self.cnn_post(image=img_spatial_cnn)['image']
        
        return img_uni, img_cnn, label


def get_dataloaders(train_dir, val_dir, batch_size=32, num_workers=4):
    """
    Create train and validation dataloaders.
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        batch_size: Batch size for both loaders
        num_workers: Number of worker processes for data loading
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = MIDOGHybridDataset(train_dir, is_train=True)
    val_dataset = MIDOGHybridDataset(val_dir, is_train=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader