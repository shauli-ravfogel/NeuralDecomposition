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

    def forward(self, X, Y, r=1e-5):
        # X,Y = torch.t(X), torch.t(Y) # X and Y are (num_dims, num_samples)

        mean_x = torch.mean(X, dim = 0)
        mean_y = torch.mean(Y, dim = 0)

        m = X.shape[1]

        X = X - mean_x
        Y = Y - mean_y

        cov_xx = (1. / (m - 1)) * torch.mm(torch.t(X), X) + r * torch.eye(m)
        cov_yy = (1. / (m - 1)) * torch.mm(torch.t(Y), Y) + r * torch.eye(m)
        cov_xy = (1. / (m - 1)) * torch.mm(torch.t(X), Y) + r * torch.eye(m)

        #print("Covariance matrix:\n", cov_xx)
        cov_xx_inverse_squared = torch.inverse(torch.cholesky(cov_xx))
        cov_yy_inverse_squared = torch.inverse(torch.cholesky(cov_yy))

        T = torch.mm(torch.mm(cov_xx_inverse_squared, cov_xy), torch.t(cov_yy_inverse_squared))
        #print("T:\n")
        #print(T, torch.min(T.view(-1)), torch.max(T.view(-1)), T.shape, T.view(-1).shape)
        U, S, V = torch.svd(T)

        A = torch.mm(cov_xx_inverse_squared, U)
        B = torch.mm(cov_yy_inverse_squared, V)

        self.A = A
        self.B = B

        X_proj = torch.mm(X, A)
        Y_proj = torch.mm(Y, B)

        return X_proj, Y_proj


if __name__ == '__main__':
    train_size = 300
    dim = 2

    cca = CCALayer()
    X = torch.rand(train_size, dim) - 0.5
    Y = 1.5 * copy.deepcopy(X)

    X_original, Y_original = copy.deepcopy(X), copy.deepcopy(Y)

    print("Performing CCA via CCA layer")

    X_cca, Y_cca = cca(X, Y)

    print(X_cca.detach().numpy()[:20, :])
    print("=========================")
    print(Y_cca.detach().numpy()[:20, :])

    print("============================================================")
    print("Performing gold CCA projection")

    gold_cca = CCA(n_components=dim)
    gold_cca.fit(X_original, Y_original)
    X_cca_true, Y_cca_true = gold_cca.transform(X_original, Y_original)
    print(X_cca_true[:20, :])
    print("=========================")
    print(Y_cca_true[:20, :])
