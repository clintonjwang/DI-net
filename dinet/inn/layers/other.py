from __future__ import annotations
from dinet.inn.fields import DiscretizedField
from dinet.inn.point_set import PointValues
import torch
nn = torch.nn
F = nn.functional
from dinet.inn import functional as inrF

class PositionalEncoding(nn.Module):
    def __init__(self, N=4, additive=True, scale=1.): #N*4 channels
        super().__init__()
        self.N = N
        self.additive = additive
        self.scale = scale
        
    def __str__(self):
        return 'PosEnc'
    def __repr__(self):
        return f"""PositionalEncoding(N={self.N})"""

    def forward(self, inr: DiscretizedField) -> DiscretizedField:
        return inrF.pos_enc(inr=inr, N=self.N, additive=self.additive, scale=self.scale)

class FlowLayer(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
    def forward(self, inr: DiscretizedField) -> DiscretizedField:
        inr.coords = inr.coords + self.layers(inr).values
        return inr
