import torch
import numpy as np
from torch import nn
import tqdm
import copy
from sklearn.cross_decomposition import CCA
import numpy as np
import scipy
from sqrtm import sqrtm

def numpy_cca(X, Y, r = 1e-5):

    print("From numpy")

    # X,Y (dim x examples)
    m = X.shape[0]

    mean_x = np.mean(X, axis = 1)
    mean_y = np.mean(Y, axis = 1)

    X = X - mean_x[:, None]
    Y = Y - mean_y[:, None]

    cov_xx = (1./(m-1)) * X.dot(X.T) + r*np.eye(m)

    cov_yy = (1. / (m - 1)) * Y.dot(Y.T) + r*np.eye(m)
    cov_xy = (1. / (m - 1)) * X.dot(Y.T)

    cov_xx_sqrt = scipy.linalg.sqrtm(cov_xx)
    cov_yy_sqrt = scipy.linalg.sqrtm(cov_yy)
    cov_xx_inverse_sqrt = np.linalg.inv(cov_xx_sqrt)
    cov_yy_inverse_sqrt = np.linalg.inv(cov_yy_sqrt)

    #print(cov_xx_inverse_sqrt.dot(cov_xx_sqrt))
    #exit()

    T = cov_xx_inverse_sqrt.dot(cov_xy).dot(cov_yy_inverse_sqrt.T)

    U,S,V =  np.linalg.svd(T)


    print("U")
    print(U)
    print("-------------------")
    print("S")
    print(S)
    print("-------------------")
    print("V")
    print(V)
    print("*********************************")


    A = cov_xx_inverse_sqrt.dot(U)
    B = cov_yy_inverse_sqrt.dot(V)
    X_proj = X.T.dot(A)
    Y_proj = Y.T.dot(B)

    return X_proj, Y_proj

class CCALayer(nn.Module):

    def __init__(self):
        super(CCALayer, self).__init__()

        """
        self.mean_x = nn.Parameter(torch.zeros(dim, requires_grad = False))
        self.mean_y = nn.Parameter(torch.zeros(dim, requires_grad = False))
        self.T = nn.Parameter(torch.zeros((final_dim, final_dim), requires_grad = False))
        self.A = nn.Parameter(torch.zeros((final_dim, final_dim), requires_grad = False))
        self.B = nn.Parameter(torch.zeros((final_dim, final_dim), requires_grad = False))
        """

    def forward(self, X, Y, r = 1e-5, is_training = True):

        X = torch.t(X)
        Y = torch.t(Y)

        torch.cuda.manual_seed_all(0)

        if is_training:

            mean_x = torch.mean(X, dim = 1, keepdim = True)
            mean_y = torch.mean(Y, dim = 1, keepdim = True)
            #self.mean_x = nn.Parameter(mean_x, requires_grad = True)
            #self.mean_y = nn.Parameter(mean_y, requires_grad = True)
            self.mean_x = mean_x
            self.mean_y = mean_y

            m = X.shape[0]

            X = X - mean_x
            Y = Y - mean_y


            cov_xx = (1. / (m - 1)) * torch.mm(X, torch.t(X)) + r * torch.eye(m).cuda()
            cov_yy = (1. / (m - 1)) * torch.mm(Y, torch.t(Y)) + r * torch.eye(m).cuda()
            cov_xy = (1. / (m - 1)) * torch.mm(X, torch.t(Y))# + r * torch.eye(m).cuda()

            #cov_xx_inverse_sqrt = torch.inverse(torch.cholesky(cov_xx, upper  = False))
            #cov_yy_inverse_sqrt = torch.inverse(torch.cholesky(cov_yy, upper = False))

            #print(cov_xx)
            #print(torch.svd(sqrtm(cov_xx)))

            cov_xx_inverse_sqrt = torch.inverse(sqrtm(cov_xx))
            cov_yy_inverse_sqrt = torch.inverse(sqrtm(cov_yy))

            T = torch.mm(torch.mm(cov_xx_inverse_sqrt, cov_xy), torch.t(cov_yy_inverse_sqrt))
            #self.T = nn.Parameter(T, requires_grad = True)
            self.T = T

            U, S, V = torch.svd(T)
            S = torch.clamp(S, 1e-7, 1 - 1e-5)
            self.S = S

            """
            print("FROM CCA LAYER")
            print("S:\n")
            print(S)
            print("U:\n")
            print(U)
            print("V:\n")
            print(V)
            print("T:\n")
            print(T)
            print("END")
            """

            A = torch.mm(cov_xx_inverse_sqrt, U)
            B = torch.mm(cov_yy_inverse_sqrt, V)
            #self.A = nn.Parameter(A, requires_grad = True)
            #self.B = nn.Parameter(B, requires_grad = True)
            self.A = A
            self.B = B

        else:

            X = X - self.mean_x
            Y = Y - self.mean_y

        X_proj = torch.mm(torch.t(X), self.A)
        Y_proj = torch.mm(torch.t(Y), self.B)

        #print(torch.mean(self.S), torch.mean(torch.diag(self.T)))
        #return torch.trace(self.T), (X_proj, Y_proj)

        #print(self.S[:100], torch.mean(self.S))
        #exit()

        return torch.mean(self.S), (X_proj, Y_proj)

if __name__ == '__main__':

    train_size = 5000
    dim = 1000

    cca = CCALayer()
    X = torch.rand(train_size, dim) - 0.5
    Y = 1.5 * copy.deepcopy(X)

    X *=1e-8
    #X = torch.zeros(train_size, dim)

    #X = torch.ones(train_size, dim)
    Y = copy.deepcopy(X) + 0.0 * (torch.rand_like(X) -0.5)

    X = torch.ones(train_size, dim)
    r = torch.range(1,train_size)
    r = r**1.4

    X = X * r[:, None] * 0.15  #+ 0.1 * ( torch.rand_like(X) - .5)
    X[:, 1] = X[:, 0]**1.5

    Y = copy.deepcopy(X) * 1.21

    X = torch.rand_like(X) - 0.5
    Y = torch.rand_like(X) - 0.5

    mean_x = torch.mean(X, dim = 0)
    mean_y = torch.mean(Y, dim = 0)

    X_original, Y_original = copy.deepcopy(X), copy.deepcopy(Y)

    #numpy_cca(X_original.detach().cpu().numpy().T, Y_original.detach().cpu().numpy().T)


    print("Performing CCA via CCA layer")

    total_corr, (X_cca, Y_cca) = cca(X.cuda(), Y.cuda())
    print(total_corr)
    print(X_cca.detach().cpu().numpy()[:20, :])
    print("=========================")
    print(Y_cca.detach().cpu().numpy()[:20, :])

    print("============================================================")
    print("Performing gold CCA projection")

    gold_cca = CCA(n_components = dim, scale = True, tol = 10-9)
    gold_cca.fit(X_original, Y_original)
    X_cca_true, Y_cca_true = gold_cca.transform(X_original, Y_original)
    print(X_cca_true[:20, :])
    print("=========================")
    print(Y_cca_true[:20, :])
    print("***********************************************")

    print("-----------------------------------------------------------------------------------")
    X_proj, Y_proj = numpy_cca(X.detach().cpu().numpy().T, Y.detach().cpu().numpy().T)
    print(X_proj[:20, :])
    print("=========================")
    print(Y_proj[:20, :])
    print(Y_proj.shape)
    print("***********************************************")
