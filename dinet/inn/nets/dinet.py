from __future__ import annotations
from dinet import inn
import pdb
from dinet.inn.support import BoundingBox
from dinet.inn.fields import FieldBatch
from dinet.inn.fields import DiscretizedField
from dinet.inn.point_set import Sampler, generate_discretization

import torch
nn = torch.nn


class DINet(nn.Module):
    def __init__(self, layers=None, encoder=(), decoder=(), sampler: dict = None):
        """DINet produces neural fields represented by an intermediate
        vector and a decoder.

        Attributes:
        - encoder: layers to produce an intermediate discretized field
        - decoder: layers to interpolate / decode the intermediate field
        - sampler: the discretization of the input nf
        """
        super().__init__()
        if layers:
            self.encoder = layers
        else:
            self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler

    def __len__(self):
        return len(self.encoder) + len(self.decoder)

    def __iter__(self):
        return iter(self.encoder) + iter(self.decoder)

    def __getitem__(self, ix):
        N = len(self.encoder)
        return self.encoder[ix] if ix < N else self.decoder[ix-N]

    def discretize_inr(self, nf: FieldBatch, sampler: Sampler | None = None) -> DiscretizedField:
        """Discretize an INRBatch into a DiscretizedINR."""
        if sampler is None:
            sampler = self.sampler
        disc = generate_discretization(domain=nf.domain, sampler=sampler)
        return DiscretizedField(disc, values=nf(disc.coords), domain=nf.domain)

    def encode(self, nf: DiscretizedField) -> DiscretizedField | torch.Tensor:
        """Produces intermediate field at discretized points"""
        return self.encoder(nf)

    def decode(self, nf: DiscretizedField, out_coords: torch.Tensor | None) -> DiscretizedField | torch.Tensor:
        """Interpolates / decodes intermediate field"""
        return self.decoder(nf, out_coords)

    def forward(self, nf: FieldBatch, out_coords: torch.Tensor | None = None) -> DiscretizedField:
        if not isinstance(nf, DiscretizedField):
            nf = self.discretize_inr(nf)
        elif not isinstance(nf, FieldBatch):
            raise ValueError(
                "nf must be INRBatch, but got {}".format(type(nf)))
        if out_coords is None:
            return self.encode(nf)
        else:
            return self.decode(self.encode(nf), out_coords)


def freeze_layer_types(dinet, classes=(inn.ChannelMixer, inn.ChannelNorm)):
    for m in dinet:
        if hasattr(m, '__iter__'):
            freeze_layer_types(m, classes)
        elif m.__class__ in classes:
            for param in m.parameters():
                param.requires_grad = False


def unfreeze_layer_types(dinet, classes=(inn.ChannelMixer, inn.ChannelNorm)):
    for m in dinet:
        if hasattr(m, '__iter__'):
            unfreeze_layer_types(m, classes)
        elif m.__class__ in classes:
            for param in m.parameters():
                param.requires_grad = True


def replace_conv_kernels(dinet, k_type='mlp', k_ratio=1.5):
    if hasattr(dinet, 'sequential'):
        return replace_conv_kernels(dinet.sequential, k_ratio=k_ratio)
    elif hasattr(dinet, 'layers'):
        return replace_conv_kernels(dinet.layers, k_ratio=k_ratio)
    length = len(dinet)
    for i in range(length):
        m = dinet[i]
        if hasattr(m, '__getitem__'):
            replace_conv_kernels(m, k_ratio=k_ratio)
        elif isinstance(m, inn.SplineConv):
            try:
                dinet[i] = replace_conv_kernel(m, k_ratio=k_ratio)
            except:
                pdb.set_trace()


def replace_conv_kernel(layer, k_type='mlp', k_ratio=1.5):
    # if k_type
    if isinstance(layer, inn.SplineConv):
        conv = inn.MLPConv(layer.in_channels, layer.out_channels,
                           kernel_support=BoundingBox.from_orthotope(
                               [k*k_ratio for k in layer.kernel_size]),
                           down_ratio=layer.down_ratio, groups=layer.groups)
        # conv.padded_extrema = layer.padded_extrema
        conv.bias = layer.bias
        return conv
    raise NotImplementedError
