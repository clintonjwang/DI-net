import os
import torch
import torchvision.models
osp = os.path
nn = torch.nn
F = nn.functional

def conv_bn_relu(in_, out_, **kwargs):
    cv = nn.Conv2d(in_, out_, 3, padding=1, bias=False, **kwargs)
    nn.init.kaiming_uniform_(cv.weight)
    return nn.Sequential(cv,
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True))

class resconv(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.layers = nn.Sequential(conv_bn_relu(C,C),conv_bn_relu(C,C))
    def forward(self, x):
        return self.layers(x)+x


def Conv2(in_channels, out_dims, C=64):
    layers = [conv_bn_relu(in_channels, C),
        # nn.MaxPool2d(2),
        conv_bn_relu(C, C*2),
        nn.AdaptiveAvgPool2d(output_size=1), nn.Flatten(1),
        nn.Linear(C*2, 128), nn.ReLU(inplace=True), nn.Linear(128, out_dims)
        # conv_bn_relu(C, C*2, stride=2),
        # nn.AdaptiveAvgPool2d(output_size=(4,4)), nn.Flatten(1),
        # nn.Linear(C*16*2, 128), nn.ReLU(inplace=True), nn.Linear(128, out_dims),
    ]
    for l in layers:
        if hasattr(l, 'weight'):
            nn.init.kaiming_uniform_(l.weight)
        if hasattr(l, 'bias'):
            nn.init.zeros_(l.bias)
    return nn.Sequential(*layers)

def Conv4(in_channels, out_dims, C=32):
    layers = [conv_bn_relu(in_channels, C),
        # nn.MaxPool2d(2),
        conv_bn_relu(C, C*2, stride=2),
        resconv(C*2),
        #conv_bn_relu(C*2, C*2),
        # nn.MaxPool2d(2),
        #conv_bn_relu(C*2, C*2),
        # conv_bn_relu(C*2, C*2),
        nn.AdaptiveAvgPool2d(output_size=(4,4)), nn.Flatten(1),
        #nn.Linear(C*2, out_dims)
        nn.Linear(C*2*16, 128), nn.ReLU(inplace=True), nn.Linear(128, out_dims),
    ]
    for l in layers:
        if hasattr(l, 'weight'):
            nn.init.kaiming_uniform_(l.weight)
        if hasattr(l, 'bias'):
            nn.init.zeros_(l.bias)
    return nn.Sequential(*layers)

class EffnetB0(nn.Module):
    def __init__(self, in_channels, out_dims):
        super().__init__()
        assert in_channels == 3
        out = nn.Linear(24, out_dims)
        nn.init.kaiming_uniform_(out.weight, mode='fan_in')
        out.bias.data.zero_()
        m = torchvision.models.efficientnet_b0(pretrained=False)
        self.layers = nn.Sequential(m.features[:3],
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(1), out)
    def forward(self, x):
        return self.layers(x)

class Resnet18(nn.Module):
    def __init__(self, in_channels, out_dims):
        super().__init__()
        assert in_channels == 3
        m = torchvision.models.resnet18(pretrained=False)
        out = nn.Linear(256, out_dims)
        nn.init.kaiming_uniform_(out.weight, mode='fan_in')
        out.bias.data.zero_()
        self.layers = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3,
            nn.AdaptiveAvgPool2d(output_size=1), nn.Flatten(1), out)
    def forward(self, x):
        return self.layers(x)

# def conv1d_bn_relu(in_, out_):
#     return nn.Sequential(nn.Conv2d(in_, out_, 3,1,1, bias=False),
#         nn.BatchNorm2d(out_),
#         nn.ReLU(inplace=True))

# def SimpleConv1d(in_channels, out_dims):
#     layers = [conv1d_bn_relu(in_channels, 32),
#         nn.MaxPool1d(2),
#         conv1d_bn_relu(32, 64),
#         nn.AdaptiveAvgPool1d(output_size=1), nn.Flatten(1),
#         nn.Linear(64, out_dims)]
#     for l in out_layers:
#         if hasattr(l, 'weight'):
#             nn.init.kaiming_uniform_(l.weight)
#         if hasattr(l, 'bias'):
#             nn.init.zeros_(l.bias)
#     return nn.Sequential(*layers)
