import contextlib
import math
from copy import deepcopy
from pathlib import Path
import torch
import torch.nn as nn
from ..modules.conv import Conv

import torch
import torch.nn as nn
from ultralytics.nn.modules import(AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,
                                    Classify, Concat, Conv, Conv2, ConvTranspose, Detect, DWConv, DWConvTranspose2d,
                                    Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, Pose, RepC3, RepConv,
                                    ResNetLayer, RTDETRDecoder, Segment, LightConv, RepConv, SpatialAttention)


import torch
import torch.nn as nn

class Conv(nn.Module):
    '''带批归一化和激活函数的卷积层'''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, bias=True, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class GSConv(nn.Module):
    """
    改进后的 GSConv 模块
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1, activation=True):
        super().__init__()
        half_channels = out_channels // 2
        self.conv1 = Conv(in_channels, half_channels, kernel_size, stride, padding=kernel_size//2, groups=groups, activation=activation)
        self.conv2 = Conv(half_channels, half_channels, 5, stride=1, padding=2, groups=half_channels, activation=activation)

    def forward(self, x):
        # 第一个卷积操作
        x1 = self.conv1(x)
        # 第二个卷积操作，并进行通道拼接
        x2 = torch.cat((x1, self.conv2(x1)), dim=1)
        # shuffle操作，通过重新排列和交换通道实现特征混合
        batch_size, channels, height, width = x2.size()
        x2 = x2.view(batch_size, 2, channels // 2, height, width)
        x2 = x2.permute(0, 2, 1, 3, 4).contiguous()
        return x2.view(batch_size, -1, height, width)

class GSConvNS(GSConv):
    """
    改进后的 GSConv 模块，带有规范化 Shuffle 操作
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1, activation=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, groups, activation)
        self.shuffle_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.act = nn.SiLU()  # 使用SiLU激活

    def forward(self, x):
        x = super().forward(x)
        # 使用额外的卷积层进行通道混合和规范化
        x = self.shuffle_conv(x)
        return self.act(x)


class GSBottleneck(nn.Module):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 1, 1),
            GSConv(c_, c2, 3, 1, act=False))
        self.shortcut = Conv(c1, c2, 1, 1, act=False)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)


class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True, use_bias=True):  # ch_in, ch_out, kernel, stride, padding, groups
        if c1 <= 0 or c2 <= 0 or k <= 0 or s <= 0:
            raise ValueError("All parameters c1, c2, k, and s must be positive integers.")
            groups = math.gcd(c1, c2)
        super().__init__(c1, c2, k, s, g=math.gcd(c1,c2),act=act, use_bias=True)

class GSBOTTleneck(GSBottleneck):

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__(c1, c2, k, s)
        self.shortcut = DWConv(c1, c2, 1, 1, act=False)



class VoVGSCSP(nn.Module):
    # VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        # self.gc1 = GSConv(c_, c_, 1, 1)
        # self.gc2 = GSConv(c_, c_, 1, 1)
        # self.gsb = GSBottleneck(c_, c_, 1, 1)
        self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, e=1.0) for _ in range(n)))
        self.res = Conv(c_, c_, 3, 1, act=False)
        self.cv3 = Conv(2 * c_, c2, 1)  #


    def forward(self, x):
        x1 = self.gsb(self.cv1(x))
        y = self.cv2(x)
        return self.cv3(torch.cat((y, x1), dim=1))


class VoVGSCSPC(VoVGSCSP):
    # cheap VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2)
        c_ = int(c2 * 0.5)  # hidden channels
        self.gsb = GSBottleneck(c_, c_, 1, 1)



