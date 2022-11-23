import torch
nn = torch.nn
from dinet.baselines.classifier import conv_bn_relu

def conv3d_bn_relu(in_, out_, k=3, **kwargs):
    cv = nn.Conv3d(in_, out_, k, padding=k//2, bias=False, **kwargs)
    nn.init.kaiming_uniform_(cv.weight)
    return nn.Sequential(cv,
        nn.BatchNorm3d(out_),
        nn.ReLU(inplace=True))

class Seg3(nn.Module):
    def __init__(self, in_channels, out_channels, C=16):
        super().__init__()
        layers = [
            conv_bn_relu(in_channels, C),
            conv_bn_relu(C, C*2),
            nn.Conv2d(C*2, out_channels, 1, bias=True),
        ]
        self.layers = nn.Sequential(*layers)
        for l in self.layers:
            if hasattr(l, 'weight'):
                nn.init.kaiming_uniform_(l.weight)
            if hasattr(l, 'bias'):
                nn.init.zeros_(l.bias)
    def forward(self, x):
        return self.layers(x)

class Seg5(nn.Module):
    def __init__(self, in_channels, out_channels, C=16):
        super().__init__()
        self.first = conv_bn_relu(in_channels, C)
        layers = [
            conv_bn_relu(C, C),
            nn.MaxPool2d(2),
            conv_bn_relu(C, C),
            conv_bn_relu(C, C),
            nn.Upsample(scale_factor=(2,2), mode='nearest'),
        ]
        self.residual = nn.Sequential(*layers)
        self.last = nn.Sequential(
            nn.Conv2d(C, out_channels, 1, bias=True))

    def forward(self, x):
        x = self.first(x)
        x = x + self.residual(x)
        return self.last(x)

class Sdf3d(nn.Module):
    def __init__(self, in_channels, out_channels, C=4, final_activation='tanh'):
        super().__init__()
        layers = [
            conv3d_bn_relu(in_channels, C, k=3),
            conv3d_bn_relu(C, C, k=5),
            nn.Conv3d(C, out_channels, 1, bias=True),
        ]
        if final_activation == 'tanh':
            layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)
        for l in self.layers:
            if hasattr(l, 'weight'):
                nn.init.kaiming_uniform_(l.weight)
            if hasattr(l, 'bias'):
                nn.init.zeros_(l.bias)
    def forward(self, x):
        return self.layers(x)
