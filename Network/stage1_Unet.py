from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torchsummary import summary
from torch.nn.parameter import Parameter

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
            # nn.LeakyReLU(0.3)
            nn.Sigmoid()
        )

class Stage1_UNet(nn.Module):
    def __init__(self,
                 patterns,
                 pattern_num,
                 W,
                 bilinear: bool = True,
                 base_c: int = 16):
        super(Stage1_UNet, self).__init__()
        self.in_channels = 1
        self.num_classes = 1
        self.bilinear = bilinear
        self.in_conv = DoubleConv(1, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, 1)
        self.forward_layer = nn.Conv2d(1,pattern_num,W,bias=False)
        self.forward_layer.weight = Parameter(patterns)
        self.forward_layer.weight.requires_grad=False

    def forward(self, x: torch.Tensor):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        pred_img = self.out_conv(x)
        pred_measurement = self.forward_layer(pred_img)

        mean_x = torch.mean(pred_img, dim=[0, 1, 2, 3])
        variance_x = torch.var(pred_img, dim=[0, 1, 2, 3])
        mean_y = torch.mean(pred_measurement, dim=[0, 1, 2, 3])
        variance_y = torch.var(pred_measurement, dim=[0, 1, 2, 3])

        out_x = (pred_img - mean_x) / torch.sqrt(variance_x)
        out_y = (pred_measurement - mean_y) / torch.sqrt(variance_y)

        return out_x, out_y

        # pred_measurement = pred_measurement / torch.max(pred_measurement)
        # return pred_img, pred_measurement




if __name__ == '__main__':
    a = torch.tensor([1])
    b = torch.tensor([2])



