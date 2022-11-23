import torch
from dinet.inn.fields import DiscretizedField

from dinet.inn.support import BoundingBox
nn = torch.nn
F = nn.functional

from dinet import inn

class Sequential(nn.Sequential):
    def forward(self, inr: DiscretizedField, query_coords=None) -> DiscretizedField|torch.Tensor:
        for module in self:
            inr = module(inr, query_coords=query_coords)
        return inr
    
def conv_norm_act(in_, out_, kernel_size=(.1,.1), **kwargs):
    kernel_support = BoundingBox.from_orthotope(dims=kernel_size)
    cv = inn.MLPConv(in_, out_, kernel_support=kernel_support, **kwargs)
    return Sequential(cv,
        inn.ChannelNorm(out_),
        inn.Activation(kwargs.pop("activation", "relu")),
    )

class ResBlock(nn.Module):
    def __init__(self, sequential: Sequential):
        super().__init__()
        self.sequential = sequential
    def forward(self, inr: DiscretizedField, query_coords=None) -> DiscretizedField:
        return inr + self.sequential(inr, query_coords=query_coords)
    def __getitem__(self, ix) -> nn.Module:
        return self.sequential[ix]
    def __len__(self):
        return len(self.sequential)
    def __iter__(self):
        return self.sequential.__iter__()

class ResConv(ResBlock):
    def __init__(self, C, **kwargs):
        super().__init__(conv_norm_act(C, C, **kwargs))

class ResConvCM(ResBlock):
    def __init__(self, C, kernel_size=(.3,.3), **kwargs):
        kernel_support = BoundingBox.from_orthotope(dims=kernel_size)
        cv = inn.MLPConv(C, C, kernel_support=kernel_support, **kwargs)
        super().__init__(Sequential(cv,
            inn.ChannelNorm(C, batchnorm=False),
            inn.ChannelMixer(C, C*4),
            inn.Activation(kwargs.pop("activation", "relu")),
            inn.ChannelMixer(C*4, C),
        ))

class ResConv2(ResBlock):
    def __init__(self, C, **kwargs):
        down_ratio = kwargs.pop("down_ratio", 1)
        super().__init__(Sequential(conv_norm_act(C, C, down_ratio=1, **kwargs),
            conv_norm_act(C, C, down_ratio=down_ratio, **kwargs)))
    # def get_convs(self):
    #     return self.sequential[0][0], self.sequential[1][0]
