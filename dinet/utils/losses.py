import torch

from dinet.inn.fields import DiscretizedField, FieldBatch
from dinet.inn.point_set import Discretization
nn = torch.nn
F = nn.functional

from dinet import inn

def mse_loss(pred,target):
    return (pred-target).pow(2).flatten(start_dim=1).mean(1)

def mean_iou(pred_seg, gt_seg):
    # pred_seg [B*N], gt_seg [B*N,C]
    iou_per_channel = (pred_seg & gt_seg).sum(0) / (pred_seg | gt_seg).sum(0)
    return iou_per_channel.mean()

def pixel_acc(pred_seg, gt_seg):
    return (pred_seg & gt_seg).sum() / pred_seg.size(0)
