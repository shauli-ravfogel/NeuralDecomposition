import torch
import numpy as np
from torch import nn

import cca_layer
import tqdm
import copy
from pytorch_revgrad import RevGrad

class ProjectionNetwork(nn.Module):

    def __init__(self, dim = 2048, final = 500):

        super(ProjectionNetwork, self).__init__()

        layers = []
        layers.append(nn.Linear(dim, 1024))
        #layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(1024, 512))
        #layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(512, 512))
        #layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(512, final))
        #layers.append(nn.Tanh())
        #layers.append(nn.Dropout(0.5))
        #layers.append(nn.LayerNorm(512))
        #layers.append(nn.Softsign())
        #layers.append(nn.Dropout(0.1))


        self.W = torch.nn.Parameter(0.0001 * (torch.randn(dim, 100) - 0.5))

        self.layers = nn.Sequential(*layers)
        self.cca = cca_layer.CCALayer(dim = final)

        pos_network = []
        pos_network.append(RevGrad())
        pos_network.append(nn.Linear(final, 512))
        #pos_network.append(nn.ReLU())
        pos_network.append(nn.Linear(512, 256))
        #pos_network.append(nn.ReLU())
        pos_network.append(nn.Linear(256, 128))
        #pos_network.append(nn.ReLU())
        pos_network.append(nn.Linear(128, 50))

        self.pos_net = nn.Sequential(*pos_network)

    def forward(self, X, Y):

        X_h, Y_h = self.layers(X), self.layers(Y)
        #X_h *= 1e-1
        #Y_h *= 1e-1
        #X_h, Y_h = torch.mm(X, self.W), torch.mm(Y, self.W)
        #X_h, Y_h = X, Y
        #print("X before CCA layer:\n")
        #print(X_h, X_h.shape)
        #print("Y before CCA layer:\n")
        #print(Y_h)
        total_corr, (X_projected, Y_projected) = self.cca(X_h,X_h, is_training = self.training)

        #print("X after CCA :\n")
        #print(X_projected)
        #print("Y after CCA :\n")
        #print(Y_projected)

        #if np.random.random() < 1e-1: print(X_h[0][:20])
        return total_corr, (X_projected, Y_projected), self.pos_net(X_h)

if __name__ == '__main__':

    train_size = 5000
    dim = 1024
    net = ProjectionNetwork()
    X = torch.rand(train_size, dim) - 0.5
    Y = -2.5 * copy.deepcopy(X)

    X_proj, Y_proj = net(X,Y)

    print(X_proj[0][:10])
    print(Y_proj[0][:10])
