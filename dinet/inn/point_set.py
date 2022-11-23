"""Point set"""
from dinet.utils import util
import itertools

import numpy as np
from dinet.inn.support import BoundingBox, Sphere, Support
from scipy.stats import qmc
import math
import torch
nn = torch.nn

PointValues = torch.Tensor
Sampler = dict

class Discretization:
    def __init__(self, coords: torch.Tensor, type: str = None, shape=None):
        """
        Args:
            coords (torch.Tensor): (N,d) coordinates of points
            type (str): 'grid', 'qmc', 'masked', etc.
        """
        assert isinstance(
            coords, torch.Tensor), "coords must be torch.Tensor, but got {}".format(type(coords))
        self.coords = coords
        self.type = type
        if shape is not None:
            self.shape = shape

    def __getattr__(self, attr):
        if attr == 'type':
            return self.type
        elif attr == 'coords':
            return self.coords
        else:
            return getattr(self.coords, attr)

    @property
    def N(self):
        return self.coords.size(-2)

    @property
    def dims(self):
        return self.coords.size(-1)


def get_sampler_from_args(dl_args, extra_args=None, c2f: bool = True):
    if isinstance(dl_args, str):
        dl_args = {"type": dl_args, **extra_args}
    kwargs = {'c2f': c2f}
    for kw in ['sample points', 'discretization type', 'discretization', 'type']:
        if kw in dl_args:
            kwargs[kw] = dl_args[kw]
    if 'image shape' in dl_args:
        kwargs['dims'] = dl_args['image shape']
    return kwargs

def get_discretizations_for_args(args) -> Discretization:
    dl_args = args["data loading"]
    if "domain" in dl_args:
        if dl_args["domain"]["type"] != "orthotope":
            raise NotImplementedError("Cannot handle non-orthotope domains")
        D = len(dl_args["domain"]["dims"])
        if dl_args["domain"]["dims"][0] == 2:
            domain = BoundingBox(*[(-1, 1)]*D)
        elif dl_args["domain"]["dims"][0] == 1:
            domain = BoundingBox(*[(0, 1)]*D)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError("No domain specified")

    c2f = util.is_model_dinet(args)
    in_args = dl_args['input discretization'] if 'input discretization' in dl_args else dl_args['discretization']
    ret = {
        'input': generate_discretization(domain=domain,
                    sampler=get_sampler_from_args(in_args, dl_args, c2f=c2f)),
    }
    if 'output discretization' in dl_args:
        ret['output'] = generate_discretization(domain=domain,
                    sampler=get_sampler_from_args(dl_args['output discretization'], dl_args, c2f=c2f))
    else:
        ret['output'] = ret['input']

    if 'test input discretization' in dl_args:
        ret['test_in'] = generate_discretization(domain=domain,
                    sampler=get_sampler_from_args(dl_args['test input discretization'], dl_args, c2f=c2f))
    else:
        ret['test_in'] = ret['input']

    if 'test output discretization' in dl_args:
        ret['test_out'] = generate_discretization(domain=domain,
                    sampler=get_sampler_from_args(dl_args['test output discretization'], dl_args, c2f=c2f))
    else:
        ret['test_out'] = ret['output']

    return ret


def generate_discretization(domain: Support, sampler: Sampler, **kwargs) -> Discretization:
    """Generates sample points for integrating along the INR

    Args:
        domain (Support): domain of INR to sample
        sampler (Sampler): specifies how to sample points

    Returns:
        Discretization: coordinates to sample
    """
    for kw in ['type', 'discretization type', 'discretization']:
        if kw in sampler:
            method = sampler[kw]
            break
        
    try:
        method;
    except:
        raise ValueError("sampler must specify discretization type, got only {}".format(sampler))
    
    shape = None
    if not isinstance(method, str):
        method = np.random.choice(method)

    if method == "grid":
        coords = meshgrid_coords(*sampler['dims'], c2f=sampler['c2f'])
        shape = sampler['dims']

    elif method == "grid_slice":
        coords = meshgrid_coords(*sampler['dims'], c2f=sampler['c2f']) # (N,d-1)
        coords = torch.cat((coords, torch.zeros_like(coords[...,-1:])), dim=-1) # (N,d)
        shape = sampler['dims']

    elif method == "shrunk":
        coords = gen_LD_seq_bbox(
            n=sampler['sample points'],
            bbox=domain.bounds, scramble=True)
        coords = coords * coords.abs()

    elif method in ("qmc", 'rqmc'):
        if isinstance(domain, Sphere):
            # Lambert equal area projection
            coords = gen_LD_seq_bbox(n=sampler['sample points'],
                                     bbox=((-1, 1), (-1, 1)), scramble=(method == 'rqmc'))
            theta = torch.arccos(2*coords[..., 0]-1)  # [0,pi]
            phi = 2*torch.pi*coords[..., 1]  # [0,2pi]
            coords = torch.stack((theta, phi), dim=-1)
        else:
            coords = gen_LD_seq_bbox(
                n=sampler['sample points'],
                bbox=domain.bounds, scramble=(method == 'rqmc'))

    elif method in ("mc", 'random'):
        if isinstance(domain, Sphere):
            raise NotImplementedError("MC sampling on sphere not implemented")
        elif isinstance(domain, BoundingBox):
            d = domain.dimensionality
            bbox = domain.bounds
            coords = torch.rand((sampler['sample points'], d),
                                dtype=torch.float, device="cuda")
            for i in range(d):
                coords[..., i] *= (bbox[i][1]-bbox[i][0]) + bbox[i][0]

    elif method == 'masked':
        assert 'mask' in kwargs
        coords = gen_LD_seq_bbox(
            n=sampler['sample points'],
            bbox=domain.bounds, scramble=(method == 'rqmc'))

    else:
        raise NotImplementedError("invalid method: "+method)

    return Discretization(coords, method, shape=shape)


def gen_LD_seq_bbox(n: int, bbox: tuple[tuple[float]],
                    scramble: bool = False, like=None, dtype=torch.float,
                    device="cuda") -> torch.Tensor:
    """Generates a low discrepancy point set on an orthotope.

    Args:
        n (int): number of points to generate.
        bbox (tuple, optional): bounds of domain
        scramble (bool, optional): randomized QMC. Defaults to False.
        like (optional): Defaults to None.
        dtype (optional): Defaults to torch.float.
        device (str, optional): Defaults to "cuda".

    Returns:
        torch.Tensor (n,d): coordinates of discretization
    """
    d = len(bbox)
    if math.log2(n) % 1 == 0:
        sampler = qmc.Sobol(d=d, scramble=scramble)
        sample = sampler.random_base2(m=int(math.log2(n)))
    else:
        sampler = qmc.Halton(d=d, scramble=scramble)
        sample = sampler.random(n=n)
    if like is None:
        out = torch.as_tensor(sample, dtype=dtype, device=device)
    else:
        out = torch.as_tensor(sample, dtype=like.dtype, device=like.device)

    for i in range(d):
        out[..., i] = out[..., i]*(bbox[i][1]-bbox[i][0])+bbox[i][0]

    return out


def meshgrid_coords(*dims, domain=(-1, 1), c2f=True,
                    dtype=torch.float, device="cuda") -> torch.Tensor:
    # c2f: coarse-to-fine ordering, puts points along coarser grid-points first
    tensors = [torch.linspace(
        *domain, steps=d, dtype=dtype, device=device) for d in dims]
    mgrid = torch.stack(util.meshgrid(*tensors, indexing='ij'), dim=-1)

    if c2f:
        factor = 2
        indices = [[0] for _ in range(len(dims))]
        new_indices = [[] for _ in range(len(dims))]
        steps = [d//2 for d in dims]
        ind_iters = []
        while np.any([s > 0 for s in steps]):
            for ix in range(len(steps)):
                if steps[ix] > 1 and steps[ix] % 2 == 1:
                    raise NotImplementedError(
                        'meshgrid is only working for powers of 2, got {}'.format(dims))
                    new_y_indices = [y for y in range(
                        1, dims[1]) if y not in y_indices]
                    ind_iters += list(itertools.product(x_indices,
                                      new_y_indices))
                elif steps[ix] > 0:
                    new_indices[ix] = list(steps[ix] * np.arange(1, factor, 2))
                    ind_iters += list(itertools.product(*
                                      indices[:ix], new_indices[ix], *indices[ix+1:]))

            if len(dims) == 2 and steps[0] > 0 and steps[1] > 0:
                ind_iters += list(itertools.product(*new_indices))
            elif len(dims) == 3:
                if steps[0] > 0 and steps[1] > 0:
                    ind_iters += list(itertools.product(
                        new_indices[0], new_indices[1], indices[2]))
                if steps[0] > 0 and steps[2] > 0:
                    ind_iters += list(itertools.product(
                        new_indices[0], indices[1], new_indices[2]))
                if steps[1] > 0 and steps[2] > 0:
                    ind_iters += list(itertools.product(
                        indices[0], new_indices[1], new_indices[2]))
                if steps[0] > 0 and steps[1] > 0 and steps[2] > 0:
                    ind_iters += list(itertools.product(
                        new_indices[0], new_indices[1], new_indices[2]))

            for ix in range(len(steps)):
                if steps[ix] > 0:
                    indices[ix] += new_indices[ix]
                    steps[ix] = steps[ix]//2
            factor *= 2

        flat_grid = mgrid.reshape(-1, len(dims))
        if len(dims) == 2:
            indices = torch.tensor([(0, 0), *ind_iters], device=device)
            indices = indices[:, 0]*dims[1] + indices[:, 1]
        elif len(dims) == 3:
            indices = torch.tensor([(0, 0, 0), *ind_iters], device=device)
            indices = indices[:, 0]*dims[1]*dims[2] + \
                indices[:, 1]*dims[2] + indices[:, 2]
        coords = flat_grid[indices]

    else:
        coords = mgrid.reshape(-1, len(dims))

    return coords


def generate_masked_sample_points(mask: torch.Tensor, sample_size: int,
                                  eps: float = 1/32) -> Discretization:
    """Generates a low discrepancy point set within a masked region.

    Args:
        mask (1,H,W): generated points must fall in this mask
        sample_size (int): number of points to generate
        eps (float, optional): Defaults to 1/32.

    Returns:
        Discretization at sampled coordinates
    """
    mask = mask.squeeze()
    if len(mask.shape) != 2:
        raise NotImplementedError('2D only')

    H, W = mask.shape
    fraction = (mask.sum()/torch.numel(mask)).item()
    coords = gen_LD_seq_bbox(n=int(sample_size/fraction * 1.2),
                             bbox=BoundingBox((eps, H-eps), (eps, W-eps)), scramble=True)
    coo = torch.floor(coords).long()
    bools = mask[coo[:, 0], coo[:, 1]]
    coord_subset = coords[bools]
    if coord_subset.size(0) < sample_size:
        return generate_masked_sample_points(mask, int(sample_size*1.5))
    return Discretization(coord_subset[:sample_size], "masked")


def get_low_discrepancy_sequence_ball(N, radius=1., eps=0., dtype=torch.float, device="cuda"):
    # what we really want is a Voronoi partition that minimizes the
    # difference between the smallest and largest cell volumes, and includes (0,0)
    #
    # Fibonacci lattice
    # http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/#more-3069
    indices = torch.arange(0, N).to(device=device, dtype=dtype) + eps
    R = radius*(indices/(N-1+2*eps)).sqrt() * \
        torch.sigmoid(torch.tensor(N).pow(.4))
    # shrink radius by some amount to increase Voronoi cells of outer points
    theta = torch.pi * (1 + 5**0.5) * indices
    return torch.stack((R*torch.cos(theta), R*torch.sin(theta)), dim=1)

from numpy.random import default_rng
class Sampler(nn.Module):
    """Maps a probability distribution over voxels to a sample from that distribution.

    Args:
        nn ([type]): [description]
    """
    def forward(probabilities: torch.Tensor, N: int) -> torch.Tensor:
        # probabilities (B,*dims,1)
        # N (int): number of samples to draw
        # returns (B,N)
        rng = default_rng()
        coords = rng.choice(torch.numel(probabilities), size=N,
                         replace=True, p=probabilities.flatten())
        return coords
