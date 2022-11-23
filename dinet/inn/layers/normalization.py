"""Normalization Layer"""
from typing import Optional
import torch

from dinet.inn.fields import DiscretizedField
from dinet.inn.point_set import PointValues
nn = torch.nn
F = nn.functional

from dinet.inn import functional as inrF

def translate_norm(norm):
    layer = ChannelNorm(channels=norm.num_features, batchnorm=isinstance(norm, nn.modules.batchnorm._BatchNorm),
            affine=norm.affine, momentum=0.1, track_running_stats=norm.track_running_stats)
    if norm.affine:
        layer.weight.data = norm.weight.data
        layer.bias.data = norm.bias.data
    if norm.track_running_stats:
        layer.running_mean.data = norm.running_mean.data
        layer.running_var.data = norm.running_var.data
    layer.eps = norm.eps
    return layer


class ChannelNorm(nn.Module):
    def __init__(self, channels:Optional[int]=None, batchnorm:bool=True,
            affine:bool=True, momentum:float=0.1,
            track_running_stats:bool=True, eps:float=1e-5,
            device=None, dtype=torch.float):
        """Encompasses Batch Norm and Instance Norm

        Args:
            channels (optional): Defaults to None.
            batchnorm (bool, optional): Defaults to True.
            affine (bool, optional): Defaults to True.
            momentum (float, optional): Defaults to 0.1.
            track_running_stats (bool, optional): Defaults to True.
            eps (optional): Defaults to 1e-5.
            device (optional): Defaults to None.
            dtype (optional): Defaults to torch.float.
        """
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.momentum = momentum
        self.eps = eps
        self.batchnorm = batchnorm
        if affine:
            self.bias = nn.Parameter(torch.zeros(channels, **factory_kwargs))
            self.weight = nn.Parameter(torch.ones(channels, **factory_kwargs))
        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(channels, **factory_kwargs))
            self.register_buffer('running_var', torch.ones(channels, **factory_kwargs))

    def __str__(self):
        if self.batchnorm:
            return 'BN'
        else:
            return 'LN'
    def __repr__(self):
        return f"ChannelNorm(batch={self.batchnorm}, affine={hasattr(self, 'weight')}, track_running_stats={hasattr(self,'running_mean')})"

    def inst_normalize(self, inr: DiscretizedField) -> DiscretizedField:
        if hasattr(self, "running_mean") and not self.training:
            mean = self.running_mean
            var = self.running_var
        else:
            mean = inr.values.mean(1, keepdim=True)
            var = inr.values.pow(2).mean(1, keepdim=True) - mean.pow(2)
            if hasattr(self, "running_mean"):
                with torch.no_grad():
                    self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mean.mean()
                    self.running_var = self.momentum * self.running_var + (1-self.momentum) * var.mean()
                mean = self.running_mean
                var = self.running_var

        inr.values = self.normalize(inr.values, mean, var)
        return inr

    def batch_normalize(self, inr: DiscretizedField) -> DiscretizedField:
        if hasattr(self, "running_mean") and not self.training:
            mean = self.running_mean
            var = self.running_var
        else:
            mean = inr.values.mean(dim=(0,1))
            var = inr.values.pow(2).mean(dim=(0,1)) - mean.pow(2)
            if hasattr(self, "running_mean"):
                with torch.no_grad():
                    self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mean
                    self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
                mean = self.running_mean
                var = self.running_var

        inr.values = self.normalize(inr.values, mean, var)
        return inr

    def normalize(self, values: PointValues, mean: torch.Tensor, var: torch.Tensor) -> PointValues:
        if hasattr(self, "weight"):
            return (values - mean)/(var.sqrt() + self.eps) * self.weight + self.bias
        else:
            return (values - mean)/(var.sqrt() + self.eps)

    def forward(self, inr: DiscretizedField, query_coords=None) -> DiscretizedField:
        if self.batchnorm:
            return self.batch_normalize(inr)
        else:
            return self.inst_normalize(inr)
