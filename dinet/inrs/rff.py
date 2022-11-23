import torch

from dinet.inn.support import BoundingBox
nn=torch.nn

from dinet import inn
from dinet.inn import point_set

def to_black_box(rff_list, **kwargs):
    evaluator = nn.ModuleList(rff_list).eval()
    return inn.NeuralFieldBatch(evaluator, domain=BoundingBox((-1,1),(-1,1)), **kwargs)

"""
An implementation of Gaussian Fourier feature mapping.

"Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
   https://arxiv.org/abs/2006.10739
   https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
"""
class RFFNet(nn.Module):
    def __init__(self, input_dims=2, num_feats=256, scale=30, eps=0.): #eps=1/256
        super().__init__()
        self.register_buffer('_B', torch.randn((input_dims, num_feats//2)) * scale)
        # self._B = nn.Parameter(torch.randn((input_dims, num_feats//2)) * scale)
        self.layers = nn.Sequential(
            nn.Conv1d(num_feats, num_feats, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_feats),
            nn.Conv1d(num_feats, num_feats, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_feats),
            nn.Conv1d(num_feats, num_feats, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_feats),
            nn.Conv1d(num_feats, 3, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid(),
        )
        self.eps = eps

    def coord_transform(self, coords): #x - [N,2]
        if self.training and hasattr(self, 'features'):
            return self.features

        coords = (coords.unsqueeze(0)+1)*(1-self.eps)/2
        x = coords @ self._B
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1).transpose(1,2)

        if self.training:
            self.features = x
        return x

    def forward(self, coords):
        return self.layers(self.coord_transform(coords)).transpose(1,2).squeeze() #[N,3]

def get_rff_keys():
    return NotImplemented

def fit_rff_to_img(target, total_steps):
    h,w = target.shape[1:]
    target = target.flatten(1).T.cuda()
    xy_grid = point_set.meshgrid_coords(h,w, c2f=False)
    model = RFFNet().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(total_steps):
        optimizer.zero_grad()
        generated = model(xy_grid)
        loss = nn.functional.l1_loss(generated, target)
        loss.backward()
        optimizer.step()
        # if epoch % 200 == 199:
    print("Loss %0.4f" % (loss.item()), flush=True)

    return model, loss
