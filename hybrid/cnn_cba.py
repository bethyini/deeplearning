"""
CNN Branch + Cross-Branch Attention

CNN Branch: Lightweight encoder for local feature extraction
Cross-Branch Attention (CBA): Fuses transformer and CNN features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


# CNN Branch
class CNNBranch(nn.Module):
    # in_channels = 3 (RGB images)
    # base_channels = starting num of feature channels
    # out_channels = final output channels (dimension for K, V in cross-attention)
    def __init__(self, in_channels=3, base_channels=32, out_channels=256):
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
        
        # Project CNN channels for attention (dim_kv)
        self.proj = nn.Conv2d(base_channels * 4, out_channels, kernel_size=1)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.layer1(x)  # Input: B×3×224×224 → B×32×224×224
        x = self.layer2(x)  # B×32×224×224 → B×32×112×112
        x = self.layer3(x)  # B×32×112×112 → B×64×56×56
        x = self.layer4(x)  # B×64×56×56 → B×128×28×28
        x = self.layer5(x)  # B×128×28×28 → B×128×14×14
        
        # Residual refinement
        identity = x
        x = self.refine(x) + identity
        
        x = self.proj(x)  # B×out_channels×14×14
        
        # Interpolate to match UNI2-h patch grid (16×16 for 224 input with patch_size=14)
        x = F.interpolate(x, size=(16, 16), mode="bilinear", align_corners=False)  # B×out_channels×16×16
        
        return x


# cross-Branch Attention

# single attention head
class CrossAttentionHead(nn.Module):
    def __init__(self, dim_q: int, dim_kv: int, hidden_dim: int):
        """
        dim_q = dimension of UNI transformer tokens (1536 for UNI2-h)
        dim_kv = dimension of CNN feature channels (C)
        hidden_dim = per-head hidden size (e.g., 128, 256)
        """
        super().__init__()

        # Q comes from UNI tokens
        # K, V come from CNN features
        self.W_Q = nn.Linear(dim_q, hidden_dim)
        self.W_K = nn.Linear(dim_kv, hidden_dim)
        self.W_V = nn.Linear(dim_kv, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, uni_tokens: torch.Tensor, kv: torch.Tensor):
        """
        uni_tokens : (B, T, dim_q)
            T = number of UNI tokens = 256 patch tokens + 8 reg tokens = 264

        kv : (B, HW, dim_kv)
            CNN features flattened:
            - CNN produces (B, C, 16, 16)
            - reshape to (B, 256, C)
        """
        Q = self.W_Q(uni_tokens)  # (B, T, hidden_dim)
        K = self.W_K(kv)  # (B, HW, hidden_dim)
        V = self.W_V(kv)  # (B, HW, hidden_dim)

        # Attention
        scores = (Q @ K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)  # (B, T, HW)
        alpha = torch.softmax(scores, dim=-1)  # (B, T, HW)
        out = alpha @ V  # (B, T, hidden_dim)

        return out, alpha


# multi-head cba
class CrossBranchAttention(nn.Module):
    """
    dim_q = transformer token dimension (1536 for UNI2-h)
    dim_kv = CNN feature dimension (channels out of CNN)
    n_hidden = per-head hidden dimension
    num_heads = number of attention heads (e.g., 4, 6, 8)
    """
    def __init__(self, dim_q: int, dim_kv: int, n_hidden: int, num_heads: int):
        super().__init__()
        self.W_O = nn.Linear(num_heads * n_hidden, dim_q)
        self.heads = nn.ModuleList([
            CrossAttentionHead(dim_q, dim_kv, n_hidden) 
            for _ in range(num_heads)
        ])
        # Residual + layernorm after combining heads
        self.norm = nn.LayerNorm(dim_q)
        
    def forward(self, uni_tokens: torch.Tensor, cnn_feat: torch.Tensor):
        """
        uni_tokens: (B, T, dim_q)
            B = batch size
            T = 264 tokens (256 patches + 8 reg tokens)
            dim_q = 1536

        cnn_feat: (B, C, H, W)
            Typically (B, out_channels, 16, 16)
        """
        B, C, H, W = cnn_feat.shape
    
        kv = cnn_feat.permute(0, 2, 3, 1).reshape(B, H*W, C)  # Convert (B, C, H, W) -> (B, HW, C)
        
        head_outputs = []
        head_alphas = []
        
        for head in self.heads:
            out, alpha = head(uni_tokens, kv)
            head_outputs.append(out)   # (B, T, n_hidden)
            head_alphas.append(alpha)  # (B, T, HW)
        
        concatenated = torch.cat(head_outputs, dim=-1)  # (B, T, n_hidden * num_heads)
        attn_output = self.W_O(concatenated)  # (B, T, dim_q)
        attn_output = self.norm(uni_tokens + attn_output)  # residual + layer norm
        attn_alphas = torch.stack(head_alphas, dim=1)  # stack heads: (B, num_heads, T, HW)
        
        return attn_output, attn_alphas
