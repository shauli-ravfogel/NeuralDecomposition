import torch
import numpy as np
from torch import nn

import cca_layer
import tqdm
import copy

class ProjectionNetwork(nn.Module):

    def __init__(self, dim = 2048):

        super(ProjectionNetwork, self).__init__()

        layers = []
        layers.append(nn.Linear(dim, 1024))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(1024, 512))
        #layers.append(nn.ReLU())
        #layers.append(nn.Linear(512, 512))

        self.W = torch.nn.Parameter(0.0001 * (torch.randn(dim, 100) - 0.5))

        self.layers = nn.Sequential(*layers)
        self.cca = cca_layer.CCALayer()

    def forward(self, X, Y):

        #print("---------------------------")

        #X_h = self.layers(X)
        #Y_h = self.layers(Y)

        X_h, Y_h = self.layers(X), self.layers(Y)
        #X_h, Y_h = torch.mm(X, self.W), torch.mm(Y, self.W)
        #X_h, Y_h = X, Y
        #print("X before CCA layer:\n")
        #print(X_h, X_h.shape)
        #print("Y before CCA layer:\n")
        #print(Y_h)
        X_projected, Y_projected = self.cca(X_h,Y_h, is_training = self.training)

        #print("X after CCA :\n")
        #print(X_projected)
        #print("Y after CCA :\n")
        #print(Y_projected)

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
