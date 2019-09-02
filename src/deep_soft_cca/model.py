import torch
from torch import nn
from pytorch_revgrad import RevGrad


class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            return din + torch.autograd.Variable(torch.randn(din.size()).cuda() * self.stddev)
        return din


class SkipConectionLinear(nn.Module):

    def __init__(self, in_dim, out_dim, substract = True):
        super(SkipConectionLinear, self).__init__()

        if in_dim != out_dim:
            raise Exception("in_dim must equal out_dim in skip connection.")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)
        self.substract = substract

    def forward(self, x):

            return x + self.linear(x)





class Siamese(nn.Module):

    def __init__(self, dim = 2048, final = 1024, skip_connections = True):

        super(Siamese, self).__init__()
        self.skip_connections = skip_connections
        layer_sizes = [dim, 1500, 512, final]
        layer_sizes = [dim, 3000, final]
        layers = []

        for i, (layer_dim, next_layer_dim) in enumerate(zip(layer_sizes,layer_sizes[1:])):

            layers.append(nn.BatchNorm1d(layer_dim))

            if i == 0 or (not self.skip_connections) or layer_dim != next_layer_dim:
                layers.append(nn.Linear(layer_dim, next_layer_dim, bias = True))
            else:
                layers.append(SkipConectionLinear(layer_dim, next_layer_dim))

            if i != len(layer_sizes) - 2:
                layers.append(nn.ReLU())

        #layers = [nn.Linear(dim, final, bias = False)]
        self.layers = nn.Sequential(*layers)


    def normalize(self, h):

        stds = torch.std(h, dim = 0, keepdim = True)
        means = torch.mean(h, dim = 0, keepdim = True)

        return (h - means) / stds

    def forward(self, x1, x2):

        h1, h2 = self.layers(x1), self.layers(x2)
        #h1, h2 = self.normalize(h1), self.normalize(h2)

        return h1, h2


