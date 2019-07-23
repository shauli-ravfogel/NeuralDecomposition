import torch
import numpy as np
from torch import nn

import cca_layer
import tqdm
import copy

class ProjectionNetwork(nn.Module):

    def __init__(self, dim = 1024):

        super(ProjectionNetwork, self).__init__()

        layers = []
        layers.append(nn.Linear(dim, 1024, bias=True))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(1024, 512))

        self.layers = nn.Sequential(*layers)
        self.cca = cca_layer.CCALayer()

    def forward(self, X, Y, r=1e-6):

        X, Y = self.layers(X), self.layers(Y)
        X_projected, Y_projected = self.cca(X,Y)

        return X_projected, Y_projected

if __name__ == '__main__':

    train_size = 5000
    dim = 1024
    net = ProjectionNetwork()
    X = torch.rand(train_size, dim) - 0.5
    Y = -2.5 * copy.deepcopy(X)

    X_proj, Y_proj = net(X,Y)

    print(X_proj[0][:10])
    print(Y_proj[0][:10])