import torch, pdb
nn = torch.nn
F = nn.functional

from dinet import inn

def translate_SE(discrete_se):
    sq,in_ = discrete_se.fc1.weight.shape[:2]
    cont_se = SqueezeExcitation(in_,sq) # need to change class to remove typing
    cont_se.fc1.weight.data = discrete_se.fc1.weight.data.squeeze(-1).squeeze(-1)
    cont_se.fc2.weight.data = discrete_se.fc2.weight.data.squeeze(-1).squeeze(-1)
    cont_se.fc1.bias.data = discrete_se.fc1.bias.data
    cont_se.fc2.bias.data = discrete_se.fc2.bias.data
    cont_se.activation = discrete_se.activation
    cont_se.scale_activation = discrete_se.scale_activation
    return cont_se

def translate_fire(discrete_fire, input_shape, extrema):
    squeeze = inn.translate_conv1x1(discrete_fire.squeeze)
    expand1x1 = inn.translate_conv1x1(discrete_fire.expand1x1)
    expand3x3 = inn.translate_conv2d(discrete_fire.expand3x3, input_shape, extrema)[0]
    cont_fire = Fire(squeeze, expand1x1, expand3x3)
    cont_fire.squeeze_activation = discrete_fire.squeeze_activation
    cont_fire.expand1x1_activation = discrete_fire.expand1x1_activation
    cont_fire.expand3x3_activation = discrete_fire.expand3x3_activation
    return cont_fire

class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels, squeeze_channels,
            activation=nn.ReLU, scale_activation=nn.Sigmoid):
        super().__init__()
        self.fc1 = nn.Linear(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Linear(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def forward(self, inr):
        scale = inr.values.mean(1)#.float()
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        inr.values = inr.values * self.scale_activation(scale).unsqueeze(1)#.double()
        return inr

class Fire(nn.Module):
    def __init__(self, squeeze, expand1x1, expand3x3) -> None:
        super().__init__()
        self.squeeze = squeeze
        self.expand1x1 = expand1x1
        self.expand3x3 = expand3x3
    def forward(self, inr):
        values = self.squeeze_activation(self.squeeze(inr.values))
        inr.values = torch.cat(
            [self.expand1x1_activation(self.expand1x1(values)),
            self.expand3x3_activation(self.expand3x3(values))],
            dim=1
        )
        return inr
