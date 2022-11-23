from __future__ import annotations
import monai.transforms as mtr
import matplotlib.pyplot as plt
from glob import glob
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dinet.inn.fields import DiscretizedField

import os
import torch
import numpy as np
from dinet import TMP_DIR

osp = os.path

rescale_clip = mtr.ScaleIntensityRangePercentiles(
    lower=1, upper=99, b_min=0, b_max=255, clip=True, dtype=np.uint8)
rescale_noclip = mtr.ScaleIntensityRangePercentiles(
    lower=0, upper=100, b_min=0, b_max=255, clip=False, dtype=np.uint8)


def rgb2d_tensor_to_npy(x):
    if len(x.shape) == 4:
        x = x[0]
    return x.permute(1, 2, 0).detach().cpu().numpy()


def grayscale2d_tensor_to_npy(x):
    x.squeeze_()
    if len(x.shape) == 3:
        x = x[0]
    return x.detach().cpu().numpy()


def BNc_to_npy(x, dims):
    return x[0].reshape(*dims, -1).squeeze(-1).detach().cpu().numpy()

def BNc_to_Bcdims(x, dims):
    return x.permute(0, 2, 1).reshape(x.size(0), -1, *dims)

def Bcdims_to_BNc(x):
    return x.flatten(start_dim=2).transpose(2, 1)


def is_model_dinet(args):
    return args["network"]['type'].lower().startswith('i') or args["network"]['type'].startswith('Tx')


def is_model_adaptive_dinet(args):
    return args["network"]['type'].lower().startswith('a')


def meshgrid(*tensors, indexing='ij') -> torch.Tensor:
    try:
        return torch.meshgrid(*tensors, indexing=indexing)
    except TypeError:
        return torch.meshgrid(*tensors)


def get_optimizer(model, args: dict, lr=None):
    opt_settings = args["optimizer"]
    if lr is None:
        lr = opt_settings["learning rate"]
    if 'beta1' in opt_settings:
        betas = (opt_settings['beta1'], .999)
    else:
        betas = (.9, .999)
    if opt_settings['type'].lower() == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr,
                                 weight_decay=args["optimizer"]["weight decay"], betas=betas)
    elif opt_settings['type'].lower() == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr,
                                weight_decay=args["optimizer"]["weight decay"], betas=betas)
    else:
        raise NotImplementedError


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def imshow(img):
    if isinstance(img, torch.Tensor):
        plt.imshow(rescale_noclip(
            img.detach().cpu().squeeze().permute(1, 2, 0).numpy()))
    else:
        plt.imshow(img)
    plt.axis('off')


def load_checkpoint(model, paths):
    if paths["pretrained model name"] is not None:
        init_weight_path = osp.join(TMP_DIR, paths["pretrained model name"])
        if not osp.exists(init_weight_path):
            raise ValueError(f"bad pretrained model path {init_weight_path}")

        checkpoint_sd = torch.load(init_weight_path)
        model_sd = model.state_dict()
        for k in model_sd.keys():
            if k in checkpoint_sd.keys() and checkpoint_sd[k].shape != model_sd[k].shape:
                checkpoint_sd.pop(k)

        model.load_state_dict(checkpoint_sd, strict=False)


def parse_int_or_list(x):
    # converts string to an int or list of ints
    if not isinstance(x, str):
        return x
    try:
        return int(x)
    except ValueError:
        return [int(s.strip()) for s in x.split(',')]


def parse_float_or_list(x):
    # converts string to a float or list of floats
    if not isinstance(x, str):
        return x
    try:
        return float(x)
    except ValueError:
        return [float(s.strip()) for s in x.split(',')]


def glob2(*paths):
    pattern = osp.expanduser(osp.join(*paths))
    if "*" not in pattern:
        pattern = osp.join(pattern, "*")
    return glob(pattern)


def flatten_list(collection):
    new_list = []
    for element in collection:
        new_list += list(element)
    return new_list


def format_float(x, n_decimals):
    if x == 0:
        return "0"
    elif np.isnan(x):
        return "NaN"
    if hasattr(x, "__iter__"):
        np.set_printoptions(precision=n_decimals)
        return str(np.array(x)).strip("[]")
    else:
        if n_decimals == 0:
            return ('%d' % x)
        else:
            return ('{:.%df}' % n_decimals).format(x)


def latex_mean_std(X=None, mean=None, stdev=None, n_decimals=1, percent=False, behaviour_if_singleton=None):
    if X is not None and len(X) == 1:
        mean = X[0]
        if not percent:
            return (r'{0:.%df}' % n_decimals).format(mean)
        else:
            return (r'{0:.%df}\%%' % n_decimals).format(mean*100)

    if stdev is None:
        mean = np.nanmean(X)
        stdev = np.nanstd(X)
    if not percent:
        return (r'{0:.%df}\pm {1:.%df}' % (n_decimals, n_decimals)).format(mean, stdev)
    else:
        return (r'{0:.%df}\%%\pm {1:.%df}\%%' % (n_decimals, n_decimals)).format(mean*100, stdev*100)

