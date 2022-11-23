"""1x1 Conv Layer"""
from __future__ import annotations

from dinet.inn.fields import DiscretizedField
import torch
nn = torch.nn
F = nn.functional
from functools import partial
he_init = nn.init.kaiming_uniform_

from dinet.inn import functional as inrF

def translate_conv1x1(conv: nn.modules.conv._ConvNd):
    bias = conv.bias is not None
    layer = ChannelMixer(in_channels=conv.weight.size(1), out_channels=conv.weight.size(0), bias=bias)
    layer.weight.data = conv.weight.data.view(conv.weight.size(0), -1)
    if bias:
        layer.bias.data = conv.bias.data
    return layer


class ChannelMixer(nn.Module):
    def __init__(self, in_channels, out_channels, normalized=False, bias=True, dtype=torch.float):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, dtype=dtype))
        he_init(self.weight, mode='fan_out', nonlinearity='relu')
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=dtype))
        self.normalized = normalized

    def __str__(self):
        return 'ChannelMixer'
    def __repr__(self):
        return f"""ChannelMixer(in_channels={self.in_channels}, 
            out_channels={self.out_channels}, bias={hasattr(self, 'bias')}, 
            normalized={self.normalized})"""

    def forward(self, inr: DiscretizedField, query_coords=None) -> DiscretizedField:
        if self.normalized:
            out = inr.values.matmul(torch.softmax(self.weight.T, dim=-1))
        else:
            out = inr.values.matmul(self.weight.T)
        if hasattr(self, "bias"):
            out += self.bias
        inr.values = out
        return inr
