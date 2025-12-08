"""
Shared model definitions for tasks.
"""
from curses import nonl
from re import M
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import timm.layers
import sys

from training.hybrid.uni_scratch.cnn_cba import CNNBranch, CrossBranchAttention

def load_uni2h():
    """
    Load UNI2h model from timm.
    """
    timm_kwargs = {
        'img_size': 224,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667 * 2,
        'num_classes': 0,  # feature extractor only
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked,
        'act_layer': torch.nn.SiLU,
        'reg_tokens': 8,
        'dynamic_img_size': True,
        }
    model = timm.create_model('hf-hub:MahmoodLab/UNI2-h', pretrained=True, **timm_kwargs)
    print("loaded UNI")
    return model

class UNI2hClassifier(nn.Module):
    """
    UNI feature extractor + 3 layer mlp classifier
    """
    def __init__(self, num_classes=2, uni_dim=1536, dropout=0.3):
        super().__init__()
        self.uni = load_uni2h()

        # freeze uni
        for param in self.uni.parameters():
            param.requires_grad = False
        self.uni.eval()

        # clasifier head
        self.classifier = nn.Sequential(
            nn.Linear(uni_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.uni(x)
        out = self.classifier(features)
        return out
    
    def train(self, mode=True):
        # Override the default train() to freeze the UNI model
        super().train(mode)
        self.uni.eval()
        return self
    
class HybridClassifier(nn.Module):
    """
    UNI + CNN with CBA
    """
    def __init__(self, num_classes=2, uni_dim=1536, cnn_out_channels=512, cba_hidden_dim=256, cba_num_heads=4, dropout=0.3):
        super().__init__()

        # uni branch frozen
        self.uni = load_uni2h()
        for param in self.uni.parameters():
            param.requires_grad = False
        self.uni.eval()
        self.uni_dim = uni_dim

        # cnn branch trainable
        self.cnn_branch = CNNBranch(in_channels=3, base_channels=32, out_channels=cnn_out_channels)

        # cba
        self.cba = CrossBranchAttention(dim_q=uni_dim, dim_kv=cnn_out_channels, n_hidden=cba_hidden_dim, num_heads=cba_num_heads)

        # feature fusion
        fused_dim = uni_dim + cnn_out_channels

        # classification head
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, img_uni:torch.Tensor, img_cnn:torch.Tensor):
        """
        img_uni: input for UNI branch, imagenet normalized, (B, 3, 224, 224)
        img_cnn: inpput for CNN branch, raw pixels to [0,1] (unless normalization exp), (B, 3, 224, 224)

        returns logits: (B, num_classes), and attn_weights (B, num_heads, T, HW)
        """
        # uni forward frozen
        with torch.no_grad():
            uni_features = self.uni.forward_features(img_uni)  # (B, 264, 1536)

        # cnn forward (trainable)
        cnn_features = self.cnn_branch(img_cnn) # (B, C, 16, 16)

        # cross branch attention
        refined_tokens, attn_weights = self.cba(uni_features, cnn_features)

        # global pool
        uni_pooled = refined_tokens.mean(dim=1)  # (B, 1536) single vector per image
        cnn_pooled = F.adaptive_avg_pool2d(cnn_features, 1).squeeze(-1).squeeze(-1)  # (B,C)

        # fusion and classification
        fused = torch.cat([uni_pooled, cnn_pooled], dim=1)
        logits = self.classifier(fused)

        return logits, attn_weights
    
    def train(self, mode=True):
        # Override the default train() to freeze the UNI model
        super().train(mode)
        self.uni.eval()
        return self


class ConvBlock(nn.Module):
    """
    Basic Conv-BN-GELU block
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
class ConvPoolBlock(nn.Module):
    """
    Conv-BN-GELU + MaxPool block
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class CNNClassifier(nn.Module):
    """
    CNN-only classifier
    """
    def __init__(self, in_channels=3, base_channels=64, feature_dim=512, num_classes=2, dropout=0.3):
        super().__init__()

        # encoder path: progressively downsample spatial dims
        self.layer1 = ConvBlock(in_channels, base_channels)         # (B, 64, 224, 224) 
        self.layer2 = ConvPoolBlock(base_channels, base_channels) # (B, 64, 112, 112)
        self.layer3 = ConvPoolBlock(base_channels, base_channels*2) # (B, 128, 56, 56)
        self.layer4 = ConvPoolBlock(base_channels*2, base_channels*4) # (B, 256, 28, 28)
        self.layer5 = ConvPoolBlock(base_channels*4, base_channels*4) # (B, 512, 14, 14)

        # feature refinement with residual connection
        self.refine = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels*4),
            ConvBlock(base_channels * 4, base_channels*4)
        )

        # project to feature dim
        self.proj = nn.Conv2d(base_channels*4, feature_dim, kernel_size=1)

        # global average pool + classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights for training from scratch
        """
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
        # encoder
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # refine
        identity = x
        x = self.refine(x) + identity

        # project + global average pool
        x = self.proj(x) # (B, feature_dim, H, W)
        x = F.adaptive_avg_pool2d(x, 1) # (B, feature_dim, 1, 1)
        x = x.view(x.size(0), -1)  # (B, feature_dim)   

        # classify
        logits = self.classifier(x)
        return logits



class PretrainedCNNClassifier(nn.Module):
    """
    pretrained CNN classifier using efficientnetB-3 imagenet backbone
    """

    def __init__(self, backbone='efficientnet_b3', num_classes=2, pretrained=True, dropout=0.3):
        super().__init__()

        # load pretrained backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)  # feature extractor only
        features_dim = self.backbone.num_features

        # classification head
        self.classifier = nn.Sequential(
            nn.Linear(features_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes
            ))

    def forward(self, x):
        """
        input: imagenet-normalized images, (B, 3, 224, 224)
        returns logits: (B, num_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
