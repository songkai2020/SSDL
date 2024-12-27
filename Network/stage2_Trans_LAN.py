'''
The code of stage2_LAN is modified from https://github.com/nagejacob/SpatiallyAdaptiveSSID/blob/main/network/lan.py
'''
import torch
import torch.nn as nn
from einops import rearrange
import numbers

# Channel transformer block.
class CTB(nn.Module):
    def __init__(self,channel,head = 2):
        super(CTB, self).__init__()
        self.qkv = nn.Conv2d(channel,3*channel,1,padding=0,bias=True)
        self.head = head

    def forward(self,x):
        b, c, h, w = x.shape
        x_qkv = self.qkv(x)
        q,k,v = x_qkv.chunk(3,dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.head)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.head)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.head)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.head, h=h, w=w)

        return out


class RB(nn.Module):
    def __init__(self, filters):
        super(RB, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, 1)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(filters, filters, 1)
        # self.cuca = CALayer(channel=filters)
        self.cuca = CTB(channel=filters)

    def forward(self, x):
        c0 = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        out = self.cuca(x)
        return out + c0

class NRB(nn.Module):
    def __init__(self, n, filters):
        super(NRB, self).__init__()
        nets = []
        for i in range(n):
            nets.append(RB(filters))
        self.body = nn.Sequential(*nets)
        self.tail = nn.Conv2d(filters, filters, 1)

    def forward(self, x):
        return x + self.tail(self.body(x))


class Trans_LAN(nn.Module):
    def __init__(self, blindspot, in_ch=1, out_ch=None, rbs=6):
        super(Trans_LAN, self).__init__()
        self.receptive_feild = blindspot
        assert self.receptive_feild % 2 == 1
        self.in_ch = in_ch
        self.out_ch = self.in_ch if out_ch is None else out_ch
        self.mid_ch = 64
        self.rbs = rbs

        layers = []
        layers.append(nn.Conv2d(self.in_ch, self.mid_ch, 1))
        layers.append(nn.ReLU())

        for i in range(self.receptive_feild // 2):
            layers.append(nn.Conv2d(self.mid_ch, self.mid_ch, 3, 1, 1))
            layers.append(nn.ReLU())

        layers.append(NRB(self.rbs, self.mid_ch))
        layers.append(nn.Conv2d(self.mid_ch, self.out_ch, 1))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

