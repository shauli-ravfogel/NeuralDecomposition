import torch
import numpy as np
import copy
import torch.nn.functional as F

class SimilarityLoss(torch.nn.Module):

    def __init__(self):

        super(SimilarityLoss, self).__init__()

    def forward(self, X, Y, total_corr):

        #dists = torch.max(torch.zeros(X.shape[0]).cuda(), 1 - torch.nn.functional.cosine_similarity(X,Y))
        #dists = torch.norm(X - Y, p = 2)
        dists = 1 - torch.nn.functional.cosine_similarity(X,Y)
        return (1 - total_corr) + torch.mean(dists)
        return torch.max(0, 1 -total_corr)


if __name__ == '__main__':

    train_size = 5000
    dim = 1024
    loss = SimilarityLoss()
    X = torch.rand(train_size, dim) - 0.5
    Y = -2.5 * copy.deepcopy(X) + 0.5 * torch.rand(train_size, dim)
    print(loss(X,Y))