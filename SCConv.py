
import torch
import torch.nn as nn
from ..modules.conv import Conv
import torch.nn.functional as F


class SCConv(nn.Module):
    """https://github.com/MCG-NKU/SCNet/blob/master/scnet.py"""

    def __init__(self, inplanes, planes, k=3, s=1, p=2, dilation=1, g=1, pooling_r=4):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            Conv(inplanes, planes, k=k, s=s, p=p, d=dilation, g=g, act=False))
        self.k3 = Conv(inplanes, planes, k=k, s=s, p=p, d=dilation, g=g, act=False)

        self.k4 = Conv(inplanes, planes, k=k, s=s, p=p, d=dilation, g=g, act=False)

    def forward(self, x):
        identity = x

        out = torch.sigmoid(
            torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:])))  # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out)  # k3 * sigmoid(identity + k2)
        out = self.k4(out)  # k4

        return out