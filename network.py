import torch
from torch import nn as nn
from torch.nn import functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
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

class SiameseUNet(nn.Module):
    def __init__(self, n_channels=1):
        super(SiameseUNet, self).__init__()
        
        # --- Query Encoder ---
        self.inc_q = DoubleConv(n_channels, 32)
        self.down1_q = Down(32, 64)
        self.down2_q = Down(64, 128)
        self.down3_q = Down(128, 256)
        self.down4_q = Down(256, 512) 

        # --- Template Encoder ---
        self.inc_t = DoubleConv(n_channels, 32)
        self.down1_t = Down(32, 64)
        self.down2_t = Down(64, 128)
        self.down3_t = Down(128, 256)
        self.down4_t = Down(256, 512)

        # --- Bottleneck ---
        self.bottleneck_conv = DoubleConv(512 + 512, 512)
        
        # --- Decoder ---
        self.up1 = Up(512 + 256, 256) 
        self.up2 = Up(256 + 128, 128)
        self.up3 = Up(128 + 64, 64)
        self.up4 = Up(64 + 32, 32)
        self.outc = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, template, query):
        # 1. Encode Query
        q1 = self.inc_q(query)
        q2 = self.down1_q(q1)
        q3 = self.down2_q(q2)
        q4 = self.down3_q(q3)
        q5 = self.down4_q(q4) 
        
        # 2. Encode Template
        t1 = self.inc_t(template)
        t2 = self.down1_t(t1)
        t3 = self.down2_t(t2)
        t4 = self.down3_t(t3)
        t5 = self.down4_t(t4) 
        
        # 3. Fuse (Spatial Concatenation)
        bottleneck = torch.cat([q5, t5], dim=1)
        bottleneck = self.bottleneck_conv(bottleneck)

        # 4. Decode Heatmap
        x = self.up1(bottleneck, q4)
        x = self.up2(x, q3)
        x = self.up3(x, q2)
        x = self.up4(x, q1)
        heatmap = self.outc(x)
        
        return heatmap
