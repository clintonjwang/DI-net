import torch
nn = torch.nn
F = nn.functional

def vectorize_siren(siren_batch):
    p_vecs = []
    for siren in siren_batch.evaluator:
        p_vecs.append(torch.nn.utils.parameters_to_vector(siren.parameters()).detach())
    return torch.stack(p_vecs, dim=0)

class MLPCls(nn.Module):
    def __init__(self, num_classes, C=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LazyBatchNorm1d(), nn.LazyLinear(C), nn.ReLU(),
            nn.BatchNorm1d(C), nn.Linear(C, C), nn.ReLU(),
            nn.Linear(C, num_classes))
            
    def forward(self, nf_batch):
        return self.layers(vectorize_siren(nf_batch))

    
def to_seg_siren(param_batch, siren_batch):
    for ix, pvec in enumerate(param_batch):
        siren = siren_batch.evaluator[ix]
        siren.net[4] = nn.Linear(256, 7, bias=False).cuda()
        torch.nn.utils.vector_to_parameters(pvec, siren.parameters())
    return siren_batch


from functools import partial
class HypernetSeg(nn.Module):
    def __init__(self, num_params, C=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(num_params), nn.Linear(num_params, C), nn.ReLU(),
            nn.BatchNorm1d(C), nn.Linear(C, C), nn.ReLU(),
            nn.Linear(C, num_params - 3*257 + 7*256))
            
    def forward(self, nf_batch, weight=.1):
        z = vectorize_siren(nf_batch)
        out = self.layers(z)
        residual, rest = out[:,:-7*256], out[:,-7*256:]
        params = torch.cat([z[:,:-3*257] + residual*weight, rest], dim=1)
        return partial(decomposed_siren, params=params)
        # return to_seg_siren(, nf_batch)


def decomposed_siren(coords, params):
    ix = 0
    n = 256*2
    n2 = n + 256
    x = torch.sin(30 * torch.einsum('Ni,Boi->BNo', coords, params[:,ix:ix+n].reshape(-1, 256,2)) + params[:,ix+n:ix+n2].unsqueeze(1))
    ix += n2
    n = 256*256
    n2 = n + 256
    x = torch.sin(30 * torch.einsum('BNi,Boi->BNo', x, params[:,ix:ix+n].reshape(-1, 256,256)) + params[:,ix+n:ix+n2].unsqueeze(1))
    ix += n2
    x = torch.sin(30 * torch.einsum('BNi,Boi->BNo', x, params[:,ix:ix+n].reshape(-1, 256,256)) + params[:,ix+n:ix+n2].unsqueeze(1))
    ix += n2
    x = torch.sin(30 * torch.einsum('BNi,Boi->BNo', x, params[:,ix:ix+n].reshape(-1, 256,256)) + params[:,ix+n:ix+n2].unsqueeze(1))
    ix += n2
    x = torch.einsum('BNi,Boi->BNo', x, params[:,ix:].reshape(-1, 7,256))
    return x
