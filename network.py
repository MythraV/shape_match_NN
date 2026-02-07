import torch
from torch import nn as nn
from torch.nn import functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SiameseCorrelationUNet(nn.Module):
    def __init__(self, n_channels=1):
        super(SiameseCorrelationUNet, self).__init__()
        
        # --- Shared Encoder ---
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512) 

        # --- Correlation & Fusion ---
        # We'll use a 1x1 conv to reduce dimensions before correlation if needed,
        # but let's try direct correlation first.
        self.fusion_conv = DoubleConv(512 + 1, 512) 
        
        # --- Decoder ---
        self.up1 = Up(512 + 256, 256) 
        self.up2 = Up(256 + 128, 128)
        self.up3 = Up(128 + 64, 64)
        self.up4 = Up(64 + 32, 32)
        self.outc = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, template, query):
        # 1. Shared Encoding
        q1 = self.inc(query)
        q2 = self.down1(q1)
        q3 = self.down2(q2)
        q4 = self.down3(q3)
        q5 = self.down4(q4) # (B, 512, 8, 8)

        t1 = self.inc(template)
        t2 = self.down1(t1)
        t3 = self.down2(t2)
        t4 = self.down3(t3)
        t5 = self.down4(t4) # (B, 512, 8, 8)

        # 2. Depth-wise Correlation (Siamese style)
        # We want to use t5 as a kernel for q5.
        # Since it's a batch, we'll do it sample by sample or use grouped convolution.
        B, C, H, W = q5.shape
        
        # Normalize features to get cosine similarity (optional but recommended)
        q5_norm = F.normalize(q5, p=2, dim=1)
        t5_norm = F.normalize(t5, p=2, dim=1)
        
        # Correlation: (B, 512, 8, 8) * (B, 512, 8, 8) -> (B, 1, 8, 8)
        # We sum over the channel dimension to get a response map
        correlation = torch.sum(q5_norm * t5_norm, dim=1, keepdim=True) # (B, 1, 8, 8)
        
        # 3. Fuse correlation back into query path
        # Concat the similarity map with query features
        fused = torch.cat([q5, correlation], dim=1)
        bottleneck = self.fusion_conv(fused)

        # 4. Decode
        x = self.up1(bottleneck, q4)
        x = self.up2(x, q3)
        x = self.up3(x, q2)
        x = self.up4(x, q1)
        heatmap = self.outc(x)
        
        return heatmap