import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import numpy as np
from functools import partial
from typing import Optional, Callable
from einops import rearrange
from ..modules.conv import Conv, DWConv, DSConv, RepConv, GhostConv, autopad
from ..modules.block import *
from .attention import *
from .rep_block import DiverseBranchBlock
from .kernel_warehouse import KWConv
from .dynamic_snake_conv import DySnakeConv
from .ops_dcnv3.modules import DCNv3, DCNv3_DyHead
from .shiftwise_conv import ReparamLargeKernelConv
from .mamba_vss import *
from .fadc import AdaptiveDilatedConv
from .hcfnet import PPA
from ..backbone.repvit import Conv2d_BN, RepVGGDW, SqueezeExcite
from ..backbone.rmt import RetBlock, RelPos2d
from .kan_convs import FastKANConv2DLayer, KANConv2DLayer, KALNConv2DLayer, KACNConv2DLayer, KAGNConv2DLayer
from .deconv import DEConv
from .SMPConv import SMPConv

from .orepa import *
from .RFAConv import *
from ultralytics.utils.torch_utils import make_divisible
from timm.layers import trunc_normal_
from timm.layers import CondConv2d

class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=None, groups=1, dilation=1, act=True, deformable_groups=1):
        super(DCNv2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        padding = autopad(kernel_size, padding, dilation)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        out_channels_offset_mask = (self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1])
        self.conv_offset_mask = nn.Conv2d(self.in_channels,out_channels_offset_mask,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding,bias=True,)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.reset_parameters()

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = torch.ops.torchvision.deform_conv2d(x, self.weight, offset, mask, self.bias, self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],self.dilation[0], self.dilation[1], self.groups, self.deformable_groups, True)
        x = self.bn(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, in_channels, out_channels, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        channels = int(out_channels * e)  # hidden channels
        self.cv1 = Conv(in_channels, channels, k[0], 1)
        self.cv2 = Conv(channels, out_channels, k[1], 1, g=g)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class DCNBottleneck1(Bottleneck):
    """bottleneck with DCNV2."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = DCNv2(c_, c2, k[1], 1)
class DCNBottleneck2(Bottleneck):
    """bottleneck with DCNV2."""
    def __init__(self, in_channels, out_channels, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(in_channels, out_channels, shortcut, g, k, e)
        channels = int(out_channels * e)  # hidden channels
        self.conv_lighting = nn.Sequential(
            DCNv2(in_channels, channels, 1, 1),
            DCNv2(channels, out_channels, 3, 1, act=False))
        self.shortcut = Conv(in_channels, out_channels, 1, 1, act=False)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)

class C2D(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(DCNBottleneck1(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

class C3D(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DCNBottleneck1(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))

class DCSP(nn.Module):
    # DCSP module with DCNBottleneck
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        channels = int(out_channels * e)  # hidden channels
        self.cv1 = Conv(in_channels, channels, 1, 1)
        self.cv2 = Conv(in_channels, channels, 1, 1)
        self.gsb = nn.Sequential(*(DCNBottleneck2(channels, channels, e=1.0) for _ in range(n)))
        self.res = Conv(channels, channels, 3, 1, act=False)
        self.cv3 = Conv(2 * channels, out_channels, 1)

    def forward(self, x):
        x1 = self.gsb(self.cv1(x))
        y = self.cv2(x)
        return self.cv3(torch.cat((y, x1), dim=1))







