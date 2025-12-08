"""
Shared data loading utilities
"""
import sys
sys.path.append("/orcd/data/edboyden/002/ezh/deeplearning/RandStainNA")

import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from randstainna import RandStainNA
from torch.utils.data import Dataset
import json

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

def get_transforms_cnn(is_train=True):
    """
    CNN-specific transforms for single-branch CNN models.

    Matches the CNN branch in HybridDataset:
      - shared spatial augmentations during training
      - Resize(224, 224) + ToFloat + ToTensorV2
      - no ImageNet normalization
    """
    if is_train:
        return A.Compose([
            # same spatial augs as HybridDataset.spatial_transform
            A.RandomRotate90(p=0.75),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),

            # cnn_post from HybridDataset
            A.Resize(224, 224),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ])
    else:
        # validation: no spatial augs, just cnn_post
        return A.Compose([
            A.Resize(224, 224),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ])



def get_transforms_uni(is_train=True, use_stain_aug=True):
    """
    get transforms for UNI branch
    """
    if is_train:
        aug_list = [
            A.RandomRotate90(p=0.75),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ]

        if use_stain_aug:
            aug_list.append(RandStainNATransform(
                yaml_file = "/orcd/data/edboyden/002/ezh/deeplearning/RandStainNA/CRC_LAB_randomTrue_n0.yaml",
                std_hyper = -0.3,
                probability = 1.0,
                distribution="normal",
                p=0.8
            ))
        aug_list.extend([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # image net norm
            ToTensorV2()
        ])
        return A.Compose(aug_list)

    else:  # validation
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # image net norm
            ToTensorV2()
        ])

def load_amibr_data(task, data_dir, val_ratio=0.2, random_state=42):
    """
    load AMIBR dataset and split into train/val sets
    returns train_paths, train_labels, val_paths, val_labels, class_weights
    """
    data_dir = Path(data_dir)
    atypical_dir = data_dir/'atypical'
    normal_dir = data_dir/'normal'

    paths = []
    labels = []


    # normal = 0, atypical = 1
    for p in normal_dir.glob('*.png'):
        paths.append(str(p))
        labels.append(0)
    for p in atypical_dir.glob('*.png'):
        paths.append(str(p))
        labels.append(1)
    
    paths = np.array(paths)
    labels = np.array(labels)

    # split
    train_paths, val_paths, train_labels, val_labels = train_test_split(paths, labels, test_size = val_ratio, stratify=labels, random_state=random_state)

    # compute class weights, since imbalanced
    n_samples = len(train_labels)
    n_normal = np.sum(train_labels==0)
    n_atypical = np.sum(train_labels==1)

    weight_normal = n_samples / (2 * n_normal)
    weight_atypical = n_samples / (2 * n_atypical)
    class_weights = torch.tensor([weight_normal, weight_atypical], dtype=torch.float)

    
    stats = {
        "total_samples": int(len(labels)),
        "normal_samples": int(np.sum(labels==0)),
        "atypical_samples": int(np.sum(labels==1)),
        "train_samples": {
            "total": int(len(train_labels)),
            "normal": int(np.sum(train_labels==0)),
            "atypical": int(np.sum(train_labels==1))
        },
        "val_samples": {
            "total": int(len(val_labels)),
            "normal": int(np.sum(val_labels==0)),
            "atypical": int(np.sum(val_labels==1))
        },
        "class_weights": {
            "normal": float(weight_normal),
            "atypical": float(weight_atypical)
        }
    }

    with open(f'/orcd/data/edboyden/002/ezh/deeplearning/task2_training/{task}_amibr_training_file.json', 'w') as f:
        json.dump(stats, f, indent=2)

    return train_paths, train_labels, val_paths, val_labels, class_weights

class SimpleDataset(Dataset):
    """
    Simple dataset for loading images and labels for uni only model
    """
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.paths[idx]).convert("RGB"))
        label = self.labels[idx]

        if self.transform:
            img = self.transform(image=img)['image']
        
        return img, label
    
class HybridDataset(Dataset):
    """
    dataset with dual processing for uni and cnn branch, sync spatial agumentations and branch-specific
    """
    def __init__(self, paths, labels, is_train=True, use_stain_aug=True):
        self.paths = paths
        self.labels = labels
        self.is_train = is_train
        self.use_stain_aug = use_stain_aug

        # shared spatial augmentations
        if is_train:  # training, not val
            self.spatial_transform = A.ReplayCompose([
                A.RandomRotate90(p=0.75),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ])
        else:
            self.spatial_transform = A.ReplayCompose([])  # no transforms on validation

        # uni-specific augmentations
        if is_train and use_stain_aug:
            self.uni_post = A.Compose([
                RandStainNATransform(
                    yaml_file = "/orcd/data/edboyden/002/ezh/deeplearning/RandStainNA/CRC_LAB_randomTrue_n0.yaml",
                    std_hyper = -0.3,
                    probability = 1.0,
                    distribution="normal",
                    p=0.8
                ),
                A.Resize(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # image net norm
                ToTensorV2()
            ])
        else:
            self.uni_post = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # image net norm
                ToTensorV2()
            ])

        # cnn specific augmentations
        self.cnn_post = A.Compose([
            A.Resize(224,224),
            A.ToFloat(max_value=255.0),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.paths[idx]).convert("RGB"))
        label = self.labels[idx]

        # apply shared spatial transforms and capture replay
        result = self.spatial_transform(image=img)
        img_spatial = result['image']
        replay  = result.get('replay', None)

        # replay same spatial transforms for cnn branch
        img_spatial_cnn = A.ReplayCompose.replay(replay, image=img)['image']

        # apply branch-specific postprocessing
        img_uni = self.uni_post(image=img_spatial)['image']
        img_cnn = self.cnn_post(image=img_spatial_cnn)['image']

        return img_uni, img_cnn, label
