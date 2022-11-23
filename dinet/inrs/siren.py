import numpy as np
import torch

from dinet.inn.support import BoundingBox
from dinet.inn.fields import NeuralFieldBatch
nn = torch.nn
F = nn.functional

def get_siren_keys():
    return ['net.0.linear.weight', 'net.0.linear.bias', 'net.1.linear.weight',
        'net.1.linear.bias', 'net.2.linear.weight', 'net.2.linear.bias', 'net.3.linear.weight',
        'net.3.linear.bias', 'net.4.weight', 'net.4.bias']

def batchify(siren_list, domain=None, **kwargs):
    if domain is None:
        domain = BoundingBox((-1,1),(-1,1))
    return NeuralFieldBatch(siren_list, #channels=siren_list[0].out_channels,
        domain=domain, **kwargs)

class Siren(nn.Module):
    def __init__(self, C=256, in_dims=2, out_channels=3, layers=3, outermost_linear=True, 
                 first_omega_0=30, hidden_omega_0=30., H=None,W=None):
        super().__init__()
        self.H, self.W = H,W
        self.net = [SineLayer(in_dims, C, 
                      is_first=True, omega_0=first_omega_0)]
        for i in range(layers):
            self.net.append(SineLayer(C, C, 
                                      is_first=False, omega_0=hidden_omega_0))
        if outermost_linear:
            final_linear = nn.Linear(C, out_channels)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / C) / hidden_omega_0, 
                                              np.sqrt(6 / C) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(C, out_channels, 
                                      is_first=False, omega_0=hidden_omega_0))
        self.net = nn.Sequential(*self.net)
        self.out_channels = out_channels
    

    def forward(self, coords):
        #coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        return self.net(coords)

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))



from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize
class ImageFitting(Dataset):
    def __init__(self, img):
        super().__init__()
        #img = Image.fromarray(img)
        C = img.size(1)
        transform = Compose([
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])
        img = transform(img)
        H,W = img.shape[-2:]
        self.pixels = img.permute(2,3, 0,1).view(-1, C)
        tensors = [torch.linspace(-1, 1, steps=H), torch.linspace(-1, 1, steps=W)]
        mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
        self.coords = mgrid.reshape(-1, 2)
        self.H,self.W = H,W

    def __len__(self):
        return 1
    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
        return self.coords, self.pixels
