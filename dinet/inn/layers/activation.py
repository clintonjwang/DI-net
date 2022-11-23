"""Activation Layer"""
from dinet.inn.fields import DiscretizedField
import torch
nn = torch.nn
F = nn.functional

def translate_activation(layer: nn.Module) -> nn.Module:
    if isinstance(layer, nn.ReLU):
        return Activation('relu')
    elif isinstance(layer, nn.LeakyReLU):
        return Activation('leakyrelu', negative_slope=layer.negative_slope)
    elif isinstance(layer, nn.SiLU):
        return Activation('swish')
    elif isinstance(layer, nn.GELU):
        return Activation('gelu')
    elif isinstance(layer, nn.Tanh):
        return Activation('tanh')
    else:
        raise NotImplementedError

def get_activation_layer(type: str|None) -> nn.Module:
    if type is None:
        return nn.Identity()
    return Activation(type)

class Activation(nn.Module):
    def __init__(self, type: str='ReLU', negative_slope: float=.1):
        super().__init__()
        self.type = type
        type = type.lower()
        if type == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif type == "leakyrelu":
            self.activation = nn.LeakyReLU(negative_slope, inplace=True)
        elif type == "gelu":
            self.activation = nn.GELU()
        elif type == "swish":
            self.activation = nn.SiLU(inplace=True)
        elif type == "tanh":
            self.activation = nn.Tanh()
        elif type == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            return NotImplemented
    def __str__(self):
        return self.type
    def forward(self, inr: DiscretizedField, query_coords=None) -> DiscretizedField:
        inr.values = self.activation(inr.values)
        return inr
