import torch
import numpy as np
import copy
import torch.nn.functional as F

class SimilarityLoss(torch.nn.Module):
    def __init__(self):
        super(SimilarityLoss, self).__init__()

    def forward(self, X, Y):

        X = F.normalize(X, p = 2, dim = 1)
        Y = F.normalize(Y, p=2, dim=1)
        XY = torch.mm(X, torch.t(Y))
        similarities = torch.abs(torch.diag(XY))
        differences = 1. - similarities

        return torch.sum(differences)

if __name__ == '__main__':

    train_size = 5000
    dim = 1024
    loss = SimilarityLoss()
    X = torch.rand(train_size, dim) - 0.5
    Y = -2.5 * copy.deepcopy(X)
    print(loss(X,Y))