import torch
from torch import nn
from pytorch_revgrad import RevGrad

class Siamese(nn.Module):

    def __init__(self, dim = 2048, final = 128):

        super(Siamese, self).__init__()

        layers = []
        layers.append(nn.BatchNorm1d(dim))
        layers.append(nn.Linear(dim, 1024, bias = True))
        #layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(1024))
        layers.append(nn.Linear(1024, 1024, bias=True))
        #layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(1024))
        layers.append(nn.Linear(1024, final, bias=True))
        #layers.append(nn.BatchNorm1d(final))
        self.layers = nn.Sequential(*layers)

    def normalize(self, h):

        stds = torch.std(h, dim = 0, keepdim = True)
        means = torch.mean(h, dim = 0, keepdim = True)

        return (h - means) / stds

    def forward(self, x1, x2):

        h1, h2 = self.layers(x1), self.layers(x2)
        #h1, h2 = self.normalize(h1), self.normalize(h2)

        return h1, h2


