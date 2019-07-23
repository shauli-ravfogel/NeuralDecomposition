import torch
import numpy as np
from torch import nn
import tqdm
import copy
from sklearn.cross_decomposition import CCA
import numpy as np


class CCALayer(nn.Module):
    def __init__(self):
        super(CCALayer, self).__init__()

    def forward(self, X, Y, r=1e-6):
        # X,Y = torch.t(X), torch.t(Y) # X and Y are (num_dims, num_samples)

        mean_x = torch.mean(X, dim=0)
        mean_y = torch.mean(Y, dim=0)

        m = X.shape[1]

        X -= mean_y
        Y -= mean_y

        cov_xx = (1. / (m - 1)) * torch.mm(torch.t(X), X) + r * torch.eye(m)
        cov_yy = (1. / (m - 1)) * torch.mm(torch.t(Y), Y) + r * torch.eye(m)
        cov_xy = (1. / (m - 1)) * torch.mm(torch.t(X), Y) + r * torch.eye(m)

        cov_xx_inverse_squared = torch.inverse(torch.cholesky(cov_xx))
        cov_yy_inverse_squared = torch.inverse(torch.cholesky(cov_yy))

        T = torch.mm(torch.mm(cov_xx_inverse_squared, cov_xy), torch.t(cov_yy_inverse_squared))
        U, S, V = torch.svd(T)

        A = torch.mm(cov_xx_inverse_squared, U)
        B = torch.mm(cov_yy_inverse_squared, V)

        self.A = A
        self.B = B

        X = torch.mm(X, A)
        Y = torch.mm(Y, B)

        return X, Y

