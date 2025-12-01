import torch
import torch.nn as nn
import torch.nn.functional as F

# Conv block
class ConvBlock(nn.Module):
    # in_ch = num input channels
    # out_ch = num output channels
    # kernel_size = size of conv filter
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False) # 2D conv layer
        self.bn = nn.BatchNorm2d(out_ch) # batch normalization layer
        self.act = nn.ReLU(inplace=True) # ReLU activation

    def forward(self, x):
        # conv -> batch normalization -> activation
        return self.act(self.bn(self.conv(x)))
    
# Conv + Pool block
class ConvPoolBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # max pooling layer, reduces spatial dims by half each time

    def forward(self, x):
        # conv -> max pooling
        return self.pool(self.conv(x))

# CNN Branch
class CNNBranch(nn.Module):
    # in_channels = 3 (RGB images)
    # base_channels = starting num of feature channels
    # output_dim = final output channels (should match transformer embedding dimension)
    def __init__(self, in_channels=3, base_channels=32, out_channels=128):
        super().__init__()
        
        # Encoder path
        self.layer1 = ConvBlock(in_channels, base_channels) # 224
        self.layer2 = ConvPoolBlock(base_channels, base_channels) # 112
        self.layer3 = ConvPoolBlock(base_channels, base_channels * 2) # 56
        self.layer4 = ConvPoolBlock(base_channels * 2, base_channels * 4) # 28
        self.layer5 = ConvPoolBlock(base_channels * 4, base_channels * 4) # 14
        
        # Feature refinement with residual connections (two conv blocks, doesn't change spatial dims)
        self.refine = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 4),
            ConvBlock(base_channels * 4, base_channels * 4)
        )
        
        # Project CNN channels for attention (dim_kv)
        self.proj = nn.Conv2d(base_channels * 4, out_channels, kernel_size=1)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.layer1(x) # Input: B×3×224×224 → B×32×224×224
        x = self.layer2(x) # B×32×224×224 → B×32×112×112
        x = self.layer3(x) # B×32×112×112 → B×64×56×56
        x = self.layer4(x) # B×64×56×56 → B×128×28×28
        x = self.layer5(x) # B×128×28×28 → B×128×14×14
        
        # Residual refinement
        identity = x
        x = self.refine(x) + identity
        
        x = self.proj(x)  # B×out_channels×14×14
        x = F.interpolate(x, size=(16, 16), mode="bilinear") #Bxout_channelsx16x16
        
        return x
