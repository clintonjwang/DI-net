import torch
nn = torch.nn
F = nn.functional
import torchkbnufft as tkbn

from dinet.baselines.classifier import conv_bn_relu
from dinet.inn.fields import FieldBatch, DiscretizedField
from dinet.inn import point_set

class NUFTBase(nn.Module):
    def __init__(self, out_size, sampler):
        super().__init__()
        self.out_size = out_size
        self.nufft_ob = tkbn.KbNufftAdjoint(self.out_size)
        self.sampler = sampler
        
    def discretize_field(self, nf: FieldBatch, sampler=None) -> DiscretizedField:
        if sampler is None:
            sampler = self.sampler
        disc = point_set.generate_discretization(domain=nf.domain, sampler=sampler)
        coords = disc.coords # (N, d)
        values = nf(disc.coords).transpose(1,2) # (B,C,N)
        return coords, values

    def forward(self, nf: FieldBatch):
        coords, values = self.discretize_field(nf)
        ifft = self.nufft_ob(values.to(torch.complex64), coords.transpose(0, 1))
        recon = torch.fft.fftshift(torch.fft.fft2(ifft)).abs() # (B,C,H,W)
        return self.layers(recon)

class NUFTCls(NUFTBase):
    def __init__(self, grid_size, in_channels, out_dims, sampler=None, C=64):
        super().__init__(grid_size, sampler)
        layers = [conv_bn_relu(in_channels, C),
            nn.MaxPool2d(2),
            conv_bn_relu(C, C*2),
            nn.AdaptiveAvgPool2d(output_size=1), nn.Flatten(1),
            nn.Linear(C*2, 128), nn.ReLU(inplace=True), nn.Linear(128, out_dims)
        ]
        for l in layers:
            if hasattr(l, 'weight'):
                nn.init.kaiming_uniform_(l.weight)
            if hasattr(l, 'bias'):
                nn.init.zeros_(l.bias)
        self.layers = nn.Sequential(*layers)


class NUFTSeg(NUFTBase):
    def __init__(self, grid_size, in_channels, out_channels, sampler=None, C=16):
        super().__init__(grid_size, sampler)
        layers = [
            conv_bn_relu(in_channels, C),
            conv_bn_relu(C, C*2),
            nn.Conv2d(C*2, out_channels, 1, bias=True),
        ]
        self.layers = nn.Sequential(*layers)
        for l in self.layers:
            if hasattr(l, 'weight'):
                nn.init.kaiming_uniform_(l.weight)
            if hasattr(l, 'bias'):
                nn.init.zeros_(l.bias)