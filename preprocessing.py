import albumentations as A
from albumentations.pytorch import ToTensorV2
from randstainna import RandStainNA
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

# shared spatial + blur transforms (NO stain here)
shared_transform = A.Compose([
    A.RandomRotate90(p=0.75),
    A.Flip(p=0.5),
    A.GaussianBlur(blur_limit=(1, 3), p=0.3),
    A.Resize(224, 224),
])

# UNI-specific tail (stain + normalization)
uni_tail = A.Compose([
    RandStainNA(std_hyper=-0.7, probability=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# CNN-specific tail (raw-ish)
raw_tail = A.Compose([
    # no stain norm, no extra blur
    # just convert to tensor (values in [0,1] if uint8 input)
    ToTensorV2(),
])


class MIDOGDataset(Dataset):
    def __init__(self, paths, labels, shared_transform, uni_tail, raw_tail):
        self.paths = paths
        self.labels = labels
        self.shared_transform = shared_transform
        self.uni_tail = uni_tail
        self.raw_tail = raw_tail

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.paths[idx]).convert("RGB"))

        # Apply spatial + blur once
        aug = self.shared_transform(image=img)
        img_aug = aug["image"]          # H x W x C, np.uint8

        # Duplicate after augment
        img_uni_np = img_aug.copy()
        img_raw_np = img_aug.copy()

        # Branch-specific tails
        img_uni = self.uni_tail(image=img_uni_np)["image"]   # normalized tensor
        img_raw = self.raw_tail(image=img_raw_np)["image"]   # raw-ish tensor

        label = self.labels[idx]
        return img_uni, img_raw, label
