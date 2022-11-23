"""Upsample Layer"""
import torch

from dinet.inn.fields import DiscretizedField
nn = torch.nn
F = nn.functional

from dinet.inn import point_set

def translate_upsample(layer, input_shape, extrema):
    h,w = input_shape
    extrema_dists = extrema[0][1] - extrema[0][0], extrema[1][1] - extrema[1][0]
    spacing = extrema_dists[0]/2 / (h-1), extrema_dists[1]/2 / (w-1)
    output_shape = input_shape[0]*2, input_shape[0]*2
    extrema = ((extrema[0][0], extrema[0][1]+spacing[0]),
            (extrema[1][0], extrema[1][1]+spacing[1]))
    if layer.scale_factor in [2,(2,2)]:
        return Upsample(4, mode=layer.mode, spacing=spacing), output_shape, extrema
    raise NotImplementedError


def get_new_coords(inr, layer):
    """Get coordinates to (re)introduce after an upsampling layer"""
    coords = inr.coords
    d = coords.shape[-1]
    N_in = coords.size(0)

    if hasattr(inr, 'dropped_coords'):
        N_new = round(N_in*(layer.scale-1))
        new_coords = inr.dropped_coords[:N_new]
        if inr.dropped_coords.size(0) == N_new:
            delattr(inr, 'dropped_coords')
        elif inr.dropped_coords.size(0) < N_new:
            raise ValueError('not enough dropped coords')
        else:
            inr.dropped_coords = inr.dropped_coords[N_new:]

    elif inr.discretization_type == 'grid':
        if layer.scale == 4:
            if d == 2:
                s0 = s1 = layer.spacing
                if layer.align_corners:
                    coords[:,0] -= s0/2
                    coords[:,1] -= s1/2
                    new_coords = torch.cat((
                        torch.stack((coords[:,0], coords[:,1]+s1), dim=1),
                        torch.stack((coords[:,0]+s0, coords[:,1]), dim=1),
                        torch.stack((coords[:,0]+s0, coords[:,1]+s1), dim=1),
                    ), dim=0)
                    coords[:,0] *= 2/(2+s0)
                    coords[:,1] *= 2/(2+s1)
                    new_coords[:,0] *= 2/(2+s0)
                    new_coords[:,1] *= 2/(2+s1)
                else:
                    new_coords = torch.cat((
                        torch.stack((coords[:,0], coords[:,1]+s1), dim=1),
                        torch.stack((coords[:,0]+s0, coords[:,1]), dim=1),
                        torch.stack((coords[:,0]+s0, coords[:,1]+s1), dim=1),
                    ), dim=0)
            else:
                s0,s1,s2 = layer.spacing
                if layer.align_corners:
                    raise NotImplementedError
                else:
                    new_coords = torch.cat((
                        torch.stack((coords[:,0], coords[:,1]+s1, coords[:,2]), dim=1),
                        torch.stack((coords[:,0]+s0, coords[:,1], coords[:,2]), dim=1),
                        torch.stack((coords[:,0]+s0, coords[:,1]+s1, coords[:,2]), dim=1),
                        torch.stack((coords[:,0], coords[:,1], coords[:,2]+s2), dim=1),
                        torch.stack((coords[:,0], coords[:,1]+s1, coords[:,2]+s2), dim=1),
                        torch.stack((coords[:,0]+s0, coords[:,1], coords[:,2]+s2), dim=1),
                        torch.stack((coords[:,0]+s0, coords[:,1]+s1, coords[:,2]+s2), dim=1),
                    ), dim=0)

        else:
            raise NotImplementedError

    elif inr.discretization_type == 'masked':
        raise ValueError('could not find dropped coords')

    else:
        new_coords = point_set.generate_quasirandom_sequence(n=N_in*layer.scale,
            d=coords.size(-1), like=coords)[N_in:]
            
    return new_coords

def upsample_nn(inr: DiscretizedField, layer) -> DiscretizedField:
    coords = inr.coords
    values = inr.values
    new_coords = get_new_coords(inr, layer)
    inr.coords = torch.cat((coords, new_coords), dim=0)
    if layer.mode == 'nearest':
        Diffs = ((new_coords - layer.shift).unsqueeze(1) - coords.unsqueeze(0)).norm(dim=-1)
        new_values = values[:,Diffs.min(dim=1).indices]
        inr.values = torch.cat((values, new_values), dim=1)

    elif layer.mode == 'k-NN':
        Diffs = ((inr.coords - layer.shift).unsqueeze(1) - coords.unsqueeze(0)).norm(dim=-1)
        values = values[:,Diffs.topk(k=layer.scale, dim=1, largest=False).indices].mean(2)
        inr.values = values
    
    return inr

def upsample_conv(inr: DiscretizedField, layer) -> DiscretizedField:
    coords = inr.coords
    values = inr.values
    new_coords = get_new_coords(inr, layer)

    inr.coords = torch.cat((coords, new_coords), dim=0)
    Diffs = ((inr.coords - layer.shift).unsqueeze(1) - coords.unsqueeze(0))
    mask = (Diffs[...,0].abs() < layer.kernel_size[0]) & (Diffs[...,1].abs() < layer.kernel_size[1])
    # new_values = values.unsqueeze(1).tile(mask.size(0),1,1)[:,mask]
    # new_values = new_values[:,mask].mean(2)
    Y = values[:,torch.where(mask)[1]] # flattened list of values of neighborhood points
    Diffs = Diffs[mask] # flattened tensor of diffs between center coords and neighboring points
    lens = tuple(mask.sum(1)) # number of kernel points assigned to each point
    Ysplit = Y.split(lens, dim=1) # list of values at neighborhood points
    newVals = [y.mean(1) for y in Ysplit]
    inr.values = torch.stack(newVals, dim=1)
    return inr

class Upsample(nn.Module):
    def __init__(self, scale, spacing=1e-3, mode='nearest', kernel=None,
            align_corners=False, dtype=torch.float):
        super().__init__()
        self.scale = scale
        self.mode = mode
        self.align_corners = align_corners
        self.spacing = spacing
        self.shift = spacing/2

    def forward(self, inr):
        if self.mode == 'nearest':
            return upsample_nn(inr, layer=self)
        elif self.mode == 'k-NN':
            return upsample_nn(inr, layer=self)
        elif self.mode == 'bilinear':
            return upsample_conv(inr, layer=self)
        else:
            raise ValueError('mode must be nearest, k-NN, or bilinear')
        return new_inr
