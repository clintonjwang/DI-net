import torch

from dinet import inn
from dinet.inn.nets.dinet import DINet
nn = torch.nn
F = nn.functional

class InrCls2(DINet):
    def __init__(self, in_channels, out_dims, sampler, C=64, **kwargs):
        out_layers = nn.Sequential(nn.Linear(C*2, 128), nn.ReLU(inplace=True), nn.Linear(128, out_dims))
        for l in out_layers:
            if hasattr(l, 'weight'):
                nn.init.kaiming_uniform_(l.weight)
                nn.init.zeros_(l.bias)
        layers = [
            inn.blocks.conv_norm_act(in_channels, C, kernel_size=(.05,.05), **kwargs),
            inn.blocks.conv_norm_act(C, C*2, kernel_size=(.2,.2), down_ratio=.25, **kwargs),
            inn.GlobalAvgPoolSequence(out_layers),
        ]
        super().__init__(sampler=sampler, layers=nn.Sequential(*layers))

class InrCls4(DINet):
    def __init__(self, in_channels, out_dims, sampler, C=32, **kwargs):
        super().__init__()
        out_layers = nn.Sequential(nn.Linear(C*2, 128), nn.ReLU(inplace=True), nn.Linear(128, out_dims))
        for l in out_layers:
            if hasattr(l, 'weight'):
                nn.init.kaiming_uniform_(l.weight)
                nn.init.zeros_(l.bias)
        layers = [
            inn.blocks.conv_norm_act(in_channels, C, kernel_size=(.05,.05), **kwargs),
            inn.blocks.conv_norm_act(C, C*2, kernel_size=(.2,.2), down_ratio=.25, **kwargs),
            inn.blocks.ResConv(C*2, kernel_size=(.3,.3), **kwargs),
            inn.GlobalAvgPoolSequence(out_layers),
        ]
        super().__init__(sampler=sampler, layers=nn.Sequential(*layers))
