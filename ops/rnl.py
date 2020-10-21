import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from .depthwise_conv import DepthwiseConv3d


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.W_k = nn.Conv3d(gate_channels, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.W_v1 = nn.Linear(gate_channels, gate_channels//reduction_ratio, bias=True)
        self.bn1 = nn.BatchNorm1d(gate_channels//reduction_ratio, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.W_v2 = nn.Linear(gate_channels//reduction_ratio, gate_channels, bias=True)

        nn.init.constant_(self.W_v2.weight, 0)
        nn.init.constant_(self.W_v2.bias, 0)


    def forward(self, x):
        # shape of x: [n,c,d,h,w]
        out = self.W_k(x) # shape: [n,1,d,h,w]
        out = out.view(x.size(0), -1, 1) # shape: [n,dhw,1]
        out = F.softmax(out, dim=1)
        r_x = x.reshape(x.size(0), x.size(1), -1) # shape: [n,c,dhw]
        out = torch.matmul(r_x, out) # shape: [n,c,1]
        out = out.view(x.size(0), -1) # shape: [n,c]
        out = self.W_v1(out) # shape: # shape: [n,c/r]
        out = out.unsqueeze(2)
        out = self.bn1(out)
        out = out.squeeze(2)
        out = self.relu(out)
        out = self.W_v2(out) # shape: [n,c]
        out = out.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        
        out = x + out

        return out

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, ratio, ks, sub_sample, bn_layer=True, activation='softmax', use_norm=False):
        super(_NonLocalBlockND, self).__init__()

        self.ratio = ratio
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.ks = np.array(ks)
        p = (np.array(ks)-1)//2
        p = p.tolist()
        
        conv_nd = nn.Conv3d
        bn = nn.BatchNorm3d
        
        self.activation = activation
        if activation == 'relu':
            self.relu = nn.ReLU(inplace=True)
        self.use_norm = use_norm
        
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.in_channels//self.ratio, kernel_size=1, stride=1, padding=0)
        if sum(self.ks > 1) > 0:
            self.penta = DepthwiseConv3d(in_channels=self.in_channels//self.ratio, ks=ks)
        
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.in_channels//self.ratio, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.in_channels//self.ratio, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)
            
        if sub_sample:
            self.pooling = nn.MaxPool3d(kernel_size=(1, 2, 2))



    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        self.feature_maps = x
        batch_size = x.size(0)
        self.attention_size = x.shape

        theta_x = self.theta(x) # [B, C/r, T,H,W]
        penta_x = self.penta(theta_x) if sum(self.ks > 1) > 0 else theta_x # [B, C/r, T,H,W]
            
        if self.use_norm: #nomalize with lengh of 1
            penta_x = F.normalize(penta_x, dim=1, p=2)
        if self.sub_sample:
            penta_x_s = self.pooling(penta_x)
            theta_x = self.pooling(theta_x)
        else:
            penta_x_s = penta_x
            
        theta_x = theta_x.view(batch_size, self.in_channels//self.ratio, -1) # [B, C/r, THW] or [B, C/r, THW/8] if pooling
        penta_x = penta_x.view(batch_size, self.in_channels//self.ratio, -1) # [B, C/r, THW]
        penta_x_s = penta_x_s.view(batch_size, self.in_channels//self.ratio, -1) # [B, C/r, THW] or [B, C/r, THW/8] if pooling

        if self.activation == 'relu':
            C = penta_x_s.size(2)
            f_div_C = self.relu(torch.matmul(penta_x_s.transpose(1,2), penta_x)) / C    # [B, THW, THW] or [B, THW/8, THW] if pooling
        elif self.activation == 'softmax':
            f_div_C = F.softmax(torch.matmul(penta_x_s.transpose(1,2), penta_x), dim=1) # [B, THW, THW] or [B, THW/8, THW] if pooling
        else:
            raise ValueError('activation is not allowed')
        self.attention_maps = f_div_C
        
        y = torch.matmul(theta_x, f_div_C) # [B, C/r, THW]

        y = y.view(batch_size, self.in_channels//self.ratio, *x.size()[2:])
        z = self.W(y) + x

        return z


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, ratio, ks, sub_sample):
        super(NONLocalBlock3D, self).__init__(in_channels=in_channels, ratio=ratio, ks=ks, sub_sample=sub_sample)


class NL3DWrapper(nn.Module):
    def __init__(self, block, n_segment, attr, ratio=2, ks=[3,7,7], sub_sample=False):
        super(NL3DWrapper, self).__init__()
        assert attr in ['cg+nl', 'nl', 'cg'], 'param attr in NL block is expected to be cg or nl or cg+nl'
        self.block = block
        self.attr = attr
        if 'cg' in attr:
            self.cg = ChannelGate(block.bn3.num_features, reduction_ratio=2)
        if 'nl' in attr:    
            self.nl = NONLocalBlock3D(in_channels=block.bn3.num_features, ratio=ratio, ks=ks, sub_sample=sub_sample)
        self.n_segment = n_segment

    def forward(self, x):
        x = self.block(x)

        nt, c, h, w = x.size()
        x = x.view(nt // self.n_segment, self.n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        if 'cg' in self.attr:
            x = self.cg(x)
        if 'nl' in self.attr:
            x = self.nl(x)
        x = x.transpose(1, 2).contiguous().view(nt, c, h, w)
        return x


def make_non_local(net, n_segment):
    import torchvision
    import archs
    if isinstance(net, torchvision.models.ResNet) or isinstance(net, archs.small_resnet.ResNet):
        net.layer2 = nn.Sequential(
            NL3DWrapper(net.layer2[0], n_segment, attr='cg+nl', ratio=2, ks=[3,7,7], sub_sample=False),
            net.layer2[1],
            NL3DWrapper(net.layer2[2], n_segment, attr='nl', ratio=2, ks=[3,7,7], sub_sample=False),
            net.layer2[3]
        )
        net.layer3 = nn.Sequential(
            NL3DWrapper(net.layer3[0], n_segment, attr='nl', ratio=2, ks=[3,3,3], sub_sample=False),
            net.layer3[1],
            NL3DWrapper(net.layer3[2], n_segment, attr='nl', ratio=2, ks=[3,3,3], sub_sample=False),
            net.layer3[3],
            NL3DWrapper(net.layer3[4], n_segment, attr='nl', ratio=2, ks=[3,3,3], sub_sample=False),
            net.layer3[5]
        )
    else:
        raise NotImplementedError


