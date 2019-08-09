import torch
import numpy as np
from torch import nn
import tqdm
import copy
from sklearn.cross_decomposition import CCA
import numpy as np


class CCALayer(nn.Module):

    def __init__(self, dim, final_dim):
        super(CCALayer, self).__init__()

        """
        self.mean_x = nn.Parameter(torch.zeros(dim, requires_grad = False))
        self.mean_y = nn.Parameter(torch.zeros(dim, requires_grad = False))
        self.T = nn.Parameter(torch.zeros((final_dim, final_dim), requires_grad = False))
        self.A = nn.Parameter(torch.zeros((final_dim, final_dim), requires_grad = False))
        self.B = nn.Parameter(torch.zeros((final_dim, final_dim), requires_grad = False))
        """

    def forward(self, X, Y, r = 1e-5, is_training = True):


        if is_training:

            mean_x = torch.mean(X, dim = 0)
            mean_y = torch.mean(Y, dim = 0)
            #self.mean_x = nn.Parameter(mean_x, requires_grad = True)
            #self.mean_y = nn.Parameter(mean_y, requires_grad = True)
            self.mean_x = mean_x
            self.mean_y = mean_y

            m = X.shape[1]

            X = X - mean_x
            Y = Y - mean_y

            cov_xx = (1. / (m - 1)) * torch.mm(torch.t(X), X) + r * torch.eye(m).cuda()
            cov_yy = (1. / (m - 1)) * torch.mm(torch.t(Y), Y) + r * torch.eye(m).cuda()
            cov_xy = (1. / (m - 1)) * torch.mm(torch.t(X), Y) + r * torch.eye(m).cuda()

            cov_xx_inverse_sqrt = torch.inverse(torch.cholesky(cov_xx))
            cov_yy_inverse_sqrt = torch.inverse(torch.cholesky(cov_yy))

            T = torch.mm(torch.mm(cov_xx_inverse_sqrt, cov_xy), torch.t(cov_yy_inverse_sqrt))
            #self.T = nn.Parameter(T, requires_grad = True)
            self.T = T

            U, S, V = torch.svd(T + r * torch.eye(m).cuda())
            S = torch.clamp(S, 1e-6, 0.9999)
            self.S = S
            
            A = torch.mm(cov_xx_inverse_sqrt, U)
            B = torch.mm(cov_yy_inverse_sqrt, V)
            #self.A = nn.Parameter(A, requires_grad = True)
            #self.B = nn.Parameter(B, requires_grad = True)
            self.A = A
            self.B = B

        else:

            X = X - self.mean_x
            Y = Y - self.mean_y

        X_proj = torch.mm(X, self.A)
        Y_proj = torch.mm(Y, self.B)


        return torch.mean(self.S), (X_proj, Y_proj)


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
