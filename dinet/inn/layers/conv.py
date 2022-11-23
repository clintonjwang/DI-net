"""Convolutional Layer"""
from __future__ import annotations
from typing import TYPE_CHECKING

from dinet.inn.support import BoundingBox, Support
import torch, pdb, math
import numpy as np
from dinet.inn.fields import DiscretizedField

from dinet.inn.point_set import Discretization
nn = torch.nn
F = nn.functional

from dinet.inn import point_set, functional as inrF
from scipy.interpolate import RectBivariateSpline as Spline2D

def get_kernel_size(input_shape:tuple, extrema:tuple=((-1,1),(-1,1)),
        k: tuple|int=3):
    h,w = input_shape
    if isinstance(k, int):
        k = (k,k)
    extrema_dists = [extrema[ix][1] - extrema[ix][0] for ix in range(len(extrema))]
    if h == 1 or w == 1:
        raise ValueError('input shape too small')
    spacing = extrema_dists[0] / (h-1), extrema_dists[1] / (w-1)
    return k[0] * spacing[0], k[1] * spacing[1]

def translate_conv2d(conv2d, input_shape, extrema=((-1,1),(-1,1)),
    smoothing=.05, **kwargs):
    # offset/grow so that the conv kernel goes a half pixel past the boundary
    dims = 2
    h,w = input_shape # shape of input features/image
    out_, in_, k1, k2 = conv2d.weight.shape
    extrema_dists = extrema[0][1] - extrema[0][0], extrema[1][1] - extrema[1][0]
    if h == 1 or w == 1:
        raise ValueError('input shape too small')
    spacing = extrema_dists[0] / (h-1), extrema_dists[1] / (w-1)
    K = k1 * spacing[0], k2 * spacing[1]
    # if zero_at_bounds:
    #     order = min(4,k1)
    # else:
    order = min(3,k1-1)

    if k1 > 3:
        smoothing = 0.
        # cannot handle different knot positions per channel
    if k1 % 2 == k2 % 2 == 0:
        shift = [spacing[ix]/2 for ix in range(dims)]
    else:
        shift = [0]*dims

    # if conv2d.padding != ((k1+1)//2-1, (k2+1)//2-1):
    #     raise NotImplementedError("padding")
    padded_extrema = [(extrema[i][0]-spacing[i]/2, extrema[i][1]+spacing[i]/2) for i in range(dims)]
    if conv2d.stride in [1,(1,1)]:
        down_ratio = 1.
        out_shape = input_shape
    elif conv2d.stride == (2,2):
        down_ratio = 1/(conv2d.stride[0]*conv2d.stride[1])
        out_shape = [input_shape[i]//2 for i in range(dims)]
        extrema = [(extrema[i][0], extrema[i][1]-spacing[i]) for i in range(dims)]
    else:
        raise NotImplementedError("down_ratio")

    bias = conv2d.bias is not None
    layer = SplineConv(in_*conv2d.groups, out_, order=order, smoothing=smoothing,
        init_weights=conv2d.weight.detach().cpu().numpy()*k1*k2,
        # scale up weights since we divide by the number of grid points
        groups=conv2d.groups, shift=shift,
        padded_extrema=padded_extrema,
        # N_bins=0,
        N_bins=2**math.ceil(math.log2(k1*k2)+3), #4
        kernel_support=BoundingBox.from_orthotope(K), down_ratio=down_ratio, bias=bias, **kwargs)
    if bias:
        layer.bias.data = conv2d.bias.data

    return layer, out_shape, extrema

def translate_conv3d(conv3d, input_shape, extrema=((-1,1),(-1,1),(-1,1)),
    **kwargs):
    # offset/grow so that the conv kernel goes a half pixel past the boundary
    dims = len(extrema)
    h,w,d = input_shape # shape of input features/image
    out_, in_ = conv3d.weight.shape[:2]
    k = conv3d.weight.shape[2:]
    extrema_dists = [extrema[ix][1] - extrema[ix][0] for ix in range(dims)]
    if h == 1 or w == 1 or d == 1:
        raise ValueError('input shape too small')
    spacing = [extrema_dists[ix] / (input_shape[ix]-1) for ix in range(dims)]
    K = [k[ix] * spacing[ix] for ix in range(dims)]

    # if conv2d.padding != ((k1+1)//2-1, (k2+1)//2-1):
    #     raise NotImplementedError("padding")
    if conv3d.stride in [1,(1,1,1)]:
        down_ratio = 1.
        out_shape = input_shape
    elif conv3d.stride == (2,2,2):
        down_ratio = 1/np.product(conv3d.stride)
        out_shape = [input_shape[i]//2 for i in range(dims)]
        extrema = [(extrema[i][0], extrema[i][1]-spacing[i]) for i in range(dims)]
    else:
        raise NotImplementedError("down_ratio")

    bias = conv3d.bias is not None
    layer = MLPConv(in_*conv3d.groups, out_,
        groups=conv3d.groups,
        kernel_support=BoundingBox.from_orthotope(K),
        down_ratio=down_ratio, bias=bias, **kwargs)
    if bias:
        layer.bias.data = conv3d.bias.data

    return layer, out_shape, extrema



class Conv(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_support:Support,
        down_ratio:float=1., groups:int=1, bias:bool=False, dtype=torch.float):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_support = kernel_support
        assert isinstance(kernel_support, BoundingBox), "only BoundBox supported"
        self.down_ratio = down_ratio
        self.groups = groups
        self.group_size = self.in_channels // self.groups
        self.dtype = dtype
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=dtype))
        else:
            self.bias = None

    @property
    def kernel_size(self):
        return self.kernel_support.shape

    def kernel_intersection_ratio(self, query_coords: Discretization):
        """Returns a factor by which to scale the outputs in case of zero-padding

        Args:
            query_coords (Discretization): points at which to obtain a scaling factor

        Returns:
            None or torch.Tensor: scaling factor
        """
        if not hasattr(self, 'padded_extrema'):
            return
        dist_to_boundary = (query_coords.unsqueeze(1) - self.padded_extrema.T.unsqueeze(0)).abs().amin(dim=1)
        k = self.kernel_size[0]/2, self.kernel_size[1]/2
        padding_ratio = (self.kernel_size[0] - F.relu(k[0] - dist_to_boundary[:,0])) * (
            self.kernel_size[1] - F.relu(k[1] - dist_to_boundary[:,1])) / self.kernel_size[0] / self.kernel_size[1]
        return padding_ratio

class MLPConv(Conv):
    def __init__(self, in_channels: int, out_channels: int,
            kernel_support: Support, mid_ch: tuple|int = 16,
            down_ratio: float=1., groups: int=1, bias: bool=False,
            mlp_type: str='siren', scale1=None, scale2=1,
            N_bins: int=64, dtype=torch.float, device='cuda'):
        """Convolutional layer with an MLP as the kernel

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_support (Support): support of the kernel
            mid_ch (tuple|int, optional): number of channels in the MLP.
            down_ratio (float, optional): downsampling ratio. Defaults to 1.
            groups (int, optional): number of channel groups. Defaults to 1.
            bias (bool, optional): whether to have additive bias. Defaults to False.
            mlp_type (str): 'standard' or 'siren'.
            scale1 ([type], optional): scale input.
            scale2 (int, optional): scale output.
            N_bins (int, optional): quantization of kernel values.
            dtype ([type], optional): torch.float
            device (str, optional): 'cuda'
        """        
        super().__init__(in_channels, out_channels, kernel_support=kernel_support,
            down_ratio=down_ratio, bias=bias, groups=groups, dtype=dtype)
        self.N_bins = N_bins
        input_dims = kernel_support.dimensionality
        # if N_bins is None:
        #     self.N_bins = 2**math.ceil(math.log2(K[0])+12) #2**math.ceil(math.log2(1/K[0]/K[1])+4)
        # else:
        if self.N_bins > 0:
            self.register_buffer("qmc_points", point_set.gen_LD_seq_bbox(n=self.N_bins,
                bbox=kernel_support.bounds, dtype=dtype, device=device))
            
        if scale1 is None:
            if mlp_type == 'siren':
                scale1 = [10./k for k in kernel_support.shape]
            else:
                scale1 = [.5/k for k in kernel_support.shape]
        self.register_buffer("scale1", torch.as_tensor(scale1, dtype=dtype))
        self.scale2 = scale2
        self.mlp_type = mlp_type
        
        if isinstance(mid_ch, int):
            mid_ch = [mid_ch]

        self.w1 = nn.Linear(input_dims, mid_ch[0])
        self.w1.weight.data.uniform_(-1/input_dims, 1/input_dims)
        layers = []
        for ix in range(1,len(mid_ch)):
            layers += [nn.Linear(mid_ch[ix-1], mid_ch[ix]), nn.ReLU(inplace=True)]
        self.kernel = nn.Sequential(*layers, nn.Linear(mid_ch[-1], out_channels * self.group_size))

        for k in range(0,len(self.kernel),2):
            nn.init.kaiming_uniform_(self.kernel[k].weight)
            self.kernel[k].bias.data.zero_()

    def __repr__(self) -> str:
        return f"""MLPConv(in_channels={self.in_channels}, out_channels={
        self.out_channels}, support={self.kernel_support}, bias={self.bias is not None})"""

    def forward(self, inr: DiscretizedField, query_coords=None) -> DiscretizedField:
        kwargs = dict(coord_to_weights=self.interpolate_weights,
            out_channels=self.out_channels,
            kernel_support=self.kernel_support,
            down_ratio=self.down_ratio,
            N_bins=self.N_bins, groups=self.groups,
            bias=self.bias, query_coords=query_coords)
        if hasattr(self, 'qmc_points'):
            kwargs['qmc_points'] = self.qmc_points
        return inrF.conv(inr, **kwargs)

    def interpolate_weights(self, coord_diffs: Discretization) -> torch.Tensor:
        if self.mlp_type == 'siren':
            activ = torch.sin
        else:
            activ = F.relu
        return self.kernel(activ(self.w1(coord_diffs * self.scale1))).reshape(
            coord_diffs.size(0), self.out_channels, self.group_size) * self.scale2


def fit_spline(values, K, order=3, smoothing=0, center=(0,0), dtype=torch.float):
    # K = dims of the entire B spline surface
    h,w = values.shape
    bbox = (-K[0]/2+center[0], K[0]/2+center[0], -K[1]/2+center[1], K[1]/2+center[1])
    x,y = (np.linspace(bbox[0][0]/h*(h-1), bbox[0][1]/h*(h-1), h),
           np.linspace(bbox[1][0]/w*(w-1), bbox[1][1]/w*(w-1), w))

    bs = Spline2D(x,y, values, bbox=bbox, kx=order,ky=order, s=smoothing)
    tx,ty,c = [torch.tensor(z).to(dtype=dtype) for z in bs.tck]
    h=tx.size(0)-order-1
    w=ty.size(0)-order-1
    c=c.reshape(h,w)
    return tx,ty,c


class SplineConv(Conv):
    def __init__(self, in_channels, out_channels, kernel_support: Support,
            init_weights, order=2, down_ratio=1.,
            N_bins=0, groups=1,
            padded_extrema=None, bias=False, smoothing=0., shift=(0,0),
            dtype=torch.float, device='cuda'):
        super().__init__(in_channels, out_channels, kernel_support=kernel_support,
            down_ratio=down_ratio, bias=bias, groups=groups, dtype=dtype)
        self.N_bins = N_bins
        if padded_extrema is not None:
            self.register_buffer("padded_extrema", torch.as_tensor(padded_extrema, dtype=dtype))
        self.register_buffer('shift', torch.tensor(shift, dtype=dtype))

        # fit pretrained kernel with b-spline
        h,w = init_weights.shape[2:]
        bbox = kernel_support.bounds
        x,y = (np.linspace(bbox[0][0]/h*(h-1), bbox[0][1]/h*(h-1), h),
               np.linspace(bbox[1][0]/w*(w-1), bbox[1][1]/w*(w-1), w))

        self.order = order
        C = []
        for i in range(self.group_size):
            C.append([])
            for o in range(out_channels):
                bs = Spline2D(x,y, init_weights[o,i], bbox=bbox, kx=order,ky=order, s=smoothing)
                tx,ty,c = [torch.tensor(z).to(dtype=dtype) for z in bs.tck]
                h=tx.size(0)-order-1
                w=ty.size(0)-order-1
                C[-1].append(c.reshape(h,w))
            C[-1] = torch.stack(C[-1],dim=0)

        self.C = nn.Parameter(torch.stack(C, dim=1))
        self.register_buffer("grid_points", torch.as_tensor(
            np.dstack(np.meshgrid(x,y)).reshape(-1,2), dtype=dtype))
        if N_bins > 0:
            self.register_buffer("qmc_points", point_set.gen_LD_seq_bbox(n=N_bins,
                bbox=bbox, dtype=dtype, device=device))
        self.register_buffer("Tx", tx)
        self.register_buffer("Ty", ty)

    def __repr__(self):
        return f"""SplineConv(in_channels={self.in_channels}, out_channels={
        self.out_channels}, kernel_size={np.round(self.kernel_size, decimals=3)}, bias={self.bias is not None})"""

    def forward(self, inr: DiscretizedField, query_coords=None):
        kwargs = dict(coord_to_weights=self.interpolate_weights,
            out_channels=self.out_channels,
            kernel_support=self.kernel_support,
            down_ratio=self.down_ratio,
            N_bins=self.N_bins, groups=self.groups,
            bias=self.bias, query_coords=query_coords)
        if hasattr(self, 'qmc_points'):
            kwargs['qmc_points'] = self.qmc_points
        return inrF.conv(inr, **kwargs)

    def interpolate_weights(self, coord_diffs):
        w_oi = []
        X = coord_diffs[:,0].unsqueeze(1)
        Y = coord_diffs[:,1].unsqueeze(1)
        px = py = self.order

        values, kx = (self.Tx<=X).min(dim=-1)
        values, ky = (self.Ty<=Y).min(dim=-1)
        kx -= 1
        ky -= 1
        kx[values] = self.Tx.size(-1)-px-2
        ky[values] = self.Ty.size(-1)-py-2

        in_, out_ = self.group_size, self.out_channels
        Dim = in_*out_
        Ctrl = self.C.view(Dim, *self.C.shape[-2:])
        for z in range(X.size(0)):
            D = Ctrl[:, kx[z]-px : kx[z]+1, ky[z]-py : ky[z]+1].clone()

            for r in range(1, px + 1):
                try:
                    alphax = (X[z,0] - self.Tx[kx[z]-px+1:kx[z]+1]) / (
                        self.Tx[2+kx[z]-r:2+kx[z]-r+px] - self.Tx[kx[z]-px+1:kx[z]+1])
                except RuntimeError:
                    print("input off the grid")
                    pdb.set_trace()
                for j in range(px, r - 1, -1):
                    D[:,j] = (1-alphax[j-1]) * D[:,j-1] + alphax[j-1] * D[:,j].clone()

            for r in range(1, py + 1):
                alphay = (Y[z,0] - self.Ty[ky[z]-py+1:ky[z]+1]) / (
                    self.Ty[2+ky[z]-r:2+ky[z]-r+py] - self.Ty[ky[z]-py+1:ky[z]+1])
                for j in range(py, r-1, -1):
                    D[:,px,j] = (1-alphay[j-1]) * D[:,px,j-1].clone() + alphay[j-1] * D[:,px,j].clone()
            
            w_oi.append(D[:,px,py])

        return torch.stack(w_oi).view(coord_diffs.size(0), self.out_channels, self.group_size)

