import torch
import numpy as np
import copy
import torch.nn.functional as F
from torch import nn
import random

class SoftCCALoss(torch.nn.Module):

    def __init__(self, p = 2):

        super(SoftCCALoss, self).__init__()
        self.p = p

    def forward(self, X, Y, r = 1e-4, alpha = 1e-0):

        m1 = torch.mean(X, dim=0, keepdim=True)
        m2 = torch.mean(Y, dim=0, keepdim=True)

        X = X - m1
        Y = Y - m2
        N, d = X.shape
        S11 = ((torch.t(X) @ X) / (N - 1)) + r * torch.eye(d).float()
        S22 = ((torch.t(Y)  @ Y) / (N - 1)) + r * torch.eye(d).float()
        S12 = (torch.t(Y) @ X) / (N - 1)

        corr_term = 0.5 * torch.norm(X - Y, p = "fro", dim = 1)**2
        # zero diagonals in covariances matrices

        S11 = S11 - torch.eye(S11.shape[0])
        S22 = S22 - torch.eye(S22.shape[0])

        decorrelation_term = torch.norm(S11, p = 1) + torch.norm(S22, p = 1)
        loss = corr_term.mean() + (1./d**2) * alpha * decorrelation_term
        return loss