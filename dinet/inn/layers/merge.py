from __future__ import annotations
import torch
from typing import Callable
from dinet.inn.fields import DiscretizedField

from dinet.inn.point_set import Discretization, PointValues
nn=torch.nn
F=nn.functional

class MergeLayer(nn.Module):
    def __init__(self, merge_function: Callable, interpolator=None):
        super().__init__()
        self.merge_function = merge_function
        self.interpolator = interpolator

    def forward(self, inr1: DiscretizedField, inr2: DiscretizedField) -> DiscretizedField:
        x = inr1.coords
        y = inr2.coords
        if len(x) == len(y):
            if torch.all(x == y):
                inr1.values = self.merge_function(inr1.values, inr2.values)
            else:
                x_indices = torch.sort((x[:,0]+2)*x.size(0)/2 + x[:,1]).indices
                y_indices = torch.sort((y[:,0]+2)*y.size(0)/2 + y[:,1]).indices
                inr1.coords = x[x_indices]
                if torch.allclose(self.coords, y[y_indices]):
                    inr1.values = self.merge_function(inr1.values[:,x_indices], inr2.values[:,y_indices])
                else:
                    raise ValueError('coord_conflict')
        else:
            raise NotImplementedError('coord_conflict')
        return inr1

    def change_discretization(self, mode: str='grid'):
        self.inr1.change_discretization(mode=mode)
        self.inr2.change_discretization(mode=mode)

def merge_domains(d1, d2):
    return (max(d1[0], d2[0]), min(d1[1], d2[1]))
    
def set_difference(x, y):
    combined = torch.cat((x, y))
    uniques, counts = combined.unique(return_counts=True)
    return uniques[counts == 1]
