# Code for paper:
# [Title]  - "Region-based Non-local Operation for Video Classification"
# [Author] - Guoxi Huang, Adrian G. Bors
# [Github] - https://github.com/guoxih/region-based-non-local-network

from __future__ import print_function
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np


def get_mask(in_channels, channels, ks):
    
    in_channels = int(in_channels)
    channels = int(channels)
    if len(ks) == 1:
        mask = np.zeros((int(in_channels), int(channels), int(ks[0])))
    elif len(ks) == 2:
        mask = np.zeros((int(in_channels), int(channels), int(ks[0]), int(ks[1])))
    elif len(ks) == 3:
        mask = np.zeros((int(in_channels), int(channels), int(ks[0]), int(ks[1]), int(ks[2])))
    else:
        raise Error('not implement yet')
    for _ in range(in_channels):
        mask[_, _ % channels, :, :] = 1.
    return mask


class DiagonalwiseRefactorization(nn.Module):
    def __init__(self, in_channels, ks, stride=1, groups=1):
        super(DiagonalwiseRefactorization, self).__init__()
        channels = in_channels // groups
        self.in_channels = in_channels
        self.groups = groups
        self.stride = stride
        
        p = (np.array(ks)-1)//2
        self.p = p.tolist()
        
        self.mask = nn.Parameter(torch.Tensor(get_mask(in_channels, channels, ks=ks)), requires_grad=False)
        self.weight = nn.Parameter(torch.Tensor(in_channels, channels, *ks), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.weight.data)
        self.weight.data.mul_(self.mask.data)
        if len(ks) == 1:
            self.conv = nn.functional.conv1d
        elif len(ks) == 2:
            self.conv = nn.functional.conv2d
        elif len(ks) == 3:
            self.conv = nn.functional.conv3d
        else:
            raise Error('The kernal size in DiagonalwiseRefactorization is wrong!')

    def forward(self, x):
        weight = torch.mul(self.weight, self.mask)

        x = self.conv(x, weight, bias=None, stride=self.stride, padding=self.p, groups=self.groups)
        return x


def DepthwiseConv3d(in_channels, ks=[3,7,7], stride=1):
    # Diagonalwise Refactorization
    # groups = 16
    assert isinstance(ks,list), 'param ks is expected be list type'
    groups = max(in_channels // 32, 1)
    return DiagonalwiseRefactorization(in_channels, ks, stride, groups)

def DepthwiseConv2d(in_channels, ks=[3,3], stride=1):
    # Diagonalwise Refactorization
    # groups = 16
    assert isinstance(ks,list), 'param ks is expected be list type'
    groups = max(in_channels // 32, 1)
    return DiagonalwiseRefactorization(in_channels, ks, stride, groups)

def DepthwiseConv1d(in_channels, ks=[3], stride=1):
    # Diagonalwise Refactorization
    # groups = 16
    assert isinstance(ks,list), 'param ks is expected be list type'
    groups = max(in_channels // 32, 1)
    return DiagonalwiseRefactorization(in_channels, ks, stride, groups)
