import torch
import numpy as np
import copy
import torch.nn.functional as F
from torch import nn
import random

class SoftCCALoss(torch.nn.Module):

    def __init__(self, p = 2, running_average = True):

        super(SoftCCALoss, self).__init__()
        self.p = p
        self.running_average = running_average

    def forward(self, X, Y, r = 1e-6, alpha = 1e-3, eps = 1e-7):

        m1 = torch.mean(X, dim=0, keepdim=True)
        m2 = torch.mean(Y, dim=0, keepdim=True)

        X = X - m1
        Y = Y - m2
        N, d = X.shape
        S11 = ((torch.t(X) @ X) / (N - 1)) + r * torch.eye(d).float().cuda()
        S22 = ((torch.t(Y)  @ Y) / (N - 1)) + r * torch.eye(d).float().cuda()
        #S12 = (torch.t(Y) @ X) / (N - 1)

        corr_term = 0.5 * torch.norm(X - Y, p = 2, dim = 1).mean()#**2

        # add variance penalty

        var_S1 = torch.diag(S11)
        var_S2 = torch.diag(S22)

        S11 = S11 -torch.diag(torch.diag(S11)) + torch.diag(1./(var_S1 + eps) - var_S1)
        S22 = S22 -torch.diag(torch.diag(S22)) + torch.diag(1./(var_S2 + eps) - var_S2)
        #S11 = S11 - torch.eye(S11.shape[0]).cuda()
        #S22 = S22 - torch.eye(S22.shape[0]).cuda()

        decorrelation_term = 0.5 * (torch.norm(S11, p = 1) + torch.norm(S22, p = 1))

        #print(corr_term, alpha * decorrelation_term)

        loss = corr_term + alpha * decorrelation_term # * (1./d**2)

        if np.random.random() < 0:
            print(torch.norm(X, dim = 1).mean())
            print(torch.diag(S11)[:10])
            print("----------------")
            print(S11[:,0][:10])
            print(alpha * decorrelation_term, corr_term)
            print("=============================")
        return loss