from dinet.baselines.classifier import conv_bn_relu
from dinet.inn.nets.dinet import DINet, AdaptiveDINet
from dinet.inn.fields import DiscretizedField
from dinet import inn
import torch

from dinet.inn.support import BoundingBox
nn = torch.nn
F = nn.functional


class ISdf3d(DINet):
    def __init__(self, in_channels, out_channels, C=8,
                 final_activation='tanh', **kwargs):
        encoder = [
            inn.blocks.conv_norm_act(in_channels, C, kernel_size=(.3, .3, .3),
                                     **kwargs),
            inn.blocks.conv_norm_act(
                C, C, kernel_size=(.6, .6, .6), down_ratio=.5, **kwargs),
            inn.ChannelMixer(C, C, bias=True),
        ]
        encoder = nn.Sequential(*encoder)
        kernel_support = BoundingBox.from_orthotope(dims=(.6, .6, .6))
        decoder = [
            inn.MLPConv(C, out_channels,
                        kernel_support=kernel_support, **kwargs),
        ]
        if final_activation is not None:
            decoder.append(inn.get_activation_layer(final_activation))
        decoder = inn.blocks.Sequential(*decoder)
        super().__init__(encoder=encoder, decoder=decoder)



class ISeg3(DINet):
    def __init__(self, in_channels, out_channels, sampler=None, C=16,
                 final_activation=None, **kwargs):
        layers = [
            inn.blocks.conv_norm_act(in_channels, C, kernel_size=(.025, .05),
                                     **kwargs),
            inn.blocks.conv_norm_act(C, C*2, kernel_size=(.075, .15),
                                     **kwargs),
            inn.ChannelMixer(C*2, out_channels, bias=True),
        ]
        if final_activation is not None:
            layers.append(inn.get_activation_layer(final_activation))
        layers = nn.Sequential(*layers)
        super().__init__(sampler=sampler, layers=layers)


class ISeg5(DINet):
    def __init__(self, in_channels, out_channels, sampler=None, C=16, **kwargs):
        super().__init__(sampler=sampler)
        self.first = nn.Sequential(
                inn.blocks.conv_norm_act(
                    in_channels, C, kernel_size=(.03, .03), **kwargs),
            )
        layers = [
            inn.blocks.conv_norm_act(
                C, C, kernel_size=(.06, .06), down_ratio=.5, **kwargs),
            inn.blocks.conv_norm_act(
                C, C, kernel_size=(.1, .1), down_ratio=.5, **kwargs),
            inn.Upsample(4),
            inn.blocks.conv_norm_act(C, C, kernel_size=(.06, .06), **kwargs),
        ]
        self.layers = nn.Sequential(*layers)
        self.last = nn.Sequential(
            inn.ChannelMixer(C, out_channels))

    def encode(self, inr: DiscretizedField) -> DiscretizedField | torch.Tensor:
        inr = self.first(inr)
        inr = inr + self.layers(inr.create_derived_inr())
        return self.last(inr)
