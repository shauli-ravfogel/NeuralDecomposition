import numpy as np
from numpy import dot
import scipy.linalg as la
import sklearn
from sklearn.cross_decomposition import CCA
from sklearn import decomposition
from sklearn.utils.extmath import randomized_svd
import time
import torch
from torch import nn


class CCAModel(object):
    def __init__(self, dim):

        self.mean_x, self.mean_y, self.A, self.B, self.sum_crr = None, None, None, None, None
        self.dim = dim

    def __call__(self, H1, H2=None, training=True, alternative=True, r=1e-5, noise = True):

        # H1 and H2 are featurs X num_points matrices containing samples columnwise.
        # dim is the desired dimensionality of CCA space.

        if training and (H2 is None):
            raise Exception("Expected two views in training.")

        if training:

            N = H1.shape[0]
            if noise:
                H1 = H1 + np.random.randn(*H1.shape) * 0.001
                H2 = H2 + np.random.randn(*H2.shape) * 0.001

            # Remove mean

            m1 = np.mean(H1, axis=0)
            m2 = np.mean(H2, axis=0)
            self.mean_x, self.mean_y = m1, m2

            H1 = H1 - m1[None, :]
            H2 = H2 - m2[None, :]

            H1, H2 = H1.T, H2.T

            S11 = ((H1.dot(H1.T)) / (N - 1)) + r * np.eye(H1.shape[0])  # cov_xx


            S22 = ((H2.dot(H2.T)) / (N - 1)) + r * np.eye(H1.shape[0])  # cov_yy
            S12 = H1.dot(H2.T) / (N - 1)  # cov_yx

            # calculate K11 = inverse(sqrt(S11)), K22 = inverse(sqrt(S22))

            D1, V1 = la.eigh(S11)
            D2, V2 = la.eigh(S22)

            K11 = V1.dot(np.diag(1 / np.sqrt(D1))).dot(V1.T)
            K22 = V2.dot(np.diag(1 / np.sqrt(D2))).dot(V2.T)

            # Calculate correlation matrix

            T = K22.dot(S12).dot(K11)

            # Perform SVD on correlation matrix

            if not alternative:
                U, D, V = np.linalg.svd(T)
                self.corr = np.mean(D)
                U, V = U[:self.dim, :], V[:self.dim, :]
                D = np.diag(D)  # correlation coefficiens of the canonical components

            else:

                # compute TT' and T'T (regularized)
                Tnp = K11.dot(S12).dot(K22)
                M1 = Tnp.dot(Tnp.T)
                M2 = Tnp.T.dot(Tnp)

                M1 += r * np.eye(M1.shape[0])
                M2 += r * np.eye(M2.shape[0])

                # compute eigen decomposition
                E1, V = la.eigh(M1)
                _, U = la.eigh(M2)
                D = np.sqrt(np.clip(E1, 1e-7, 1.))
                self.D = D
                self.corr = np.mean(D[-self.dim:])
                U, V = U.T[-self.dim:, :], V.T[-self.dim:, :]

            A = K11.dot(V.T)  # projection matrix for H1
            B = K22.dot(U.T)  # projection matrix for H2

            s = np.sign(np.diag(V.dot(S12).dot(U.T)))
            B *= s
            self.A, self.B = A, B

            # Project & return
            H1_proj, H2_proj = H1.T.dot(self.A), H2.T.dot(self.B)
            return H1_proj[:, ::-1], H2_proj[:, ::-1], self.corr

        else:
            # in test time, use saved mean and projection matrix.
            H1 -= self.mean_x[None, :]
            H2 -= self.mean_y[None, :]
            x_proj = (H1.dot(self.A))[:, ::-1]
            y_proj = (H2.dot(self.B))[:, ::-1]
            return x_proj, y_proj

class CCALayer(nn.Module):
    def __init__(self, dim):

        super(CCALayer, self).__init__()
        self.dim = dim

    def forward(self, H1, H2=None, is_training=True, r=1e-4, np_sqrt = False):

        # H1 and H2 are DXN matrices containing samples columnwise.
        # dim is the desired dimensionality of CCA space.

        if is_training and (H2 is None):
            raise Exception("Expected two views in training.")

        if is_training:

            N, d = H1.shape
            # Remove mean
            m1 = torch.mean(H1, dim=0, keepdim=True)
            m2 = torch.mean(H2, dim=0, keepdim=True)
            self.mean_x, self.mean_y = m1, m2

            H1 = H1 - m1
            H2 = H2 - m2

            H1,H2 = torch.t(H1), torch.t(H2)

            S11 = ((H1 @ torch.t(H1)) / (N - 1)) + r * torch.eye(d).float().cuda()


            S22 = ((H2 @ torch.t(H2)) / (N - 1)) + r * torch.eye(d).float().cuda()
            S12 = (H1 @ torch.t(H2)) / (N - 1)

            D1, V1 = torch.symeig(S11, eigenvectors=True)




            D2, V2 = torch.symeig(S22, eigenvectors=True)
            #D1 = torch.clamp(D1, min= r, max = 1.)
            #D2 = D2.clamp(D2, min = r, max = 1.)

            if np_sqrt:
                diag_sqrt_inverse_D2 = torch.diag(torch.from_numpy(
                    np.reciprocal(np.sqrt(D2.detach().cpu().numpy())))).cuda()
                diag_sqrt_inverse_D1 = torch.diag(torch.from_numpy(
                    np.reciprocal(np.sqrt(D1.detach().cpu().numpy())))).cuda()
            else:

                diag_sqrt_inverse_D2 = torch.diag(1. / torch.sqrt(D2))
                diag_sqrt_inverse_D1 = torch.diag(1. / torch.sqrt(D1))

            K11 = V1 @ diag_sqrt_inverse_D1 @ torch.t(V1)  # dot(dot(V1,np.diag(1/np.sqrt(D1))),np.transpose(V1))
            K22 = V2 @ diag_sqrt_inverse_D2 @ torch.t(V2)


            Tnp = K11 @ S12 @ K22
            M1 = Tnp @ torch.t(Tnp)
            M2 = torch.t(Tnp) @ Tnp

            M1 += r * torch.eye(M1.shape[0]).float().cuda()
            M2 += r * torch.eye(M2.shape[0]).float().cuda()

            # compute eigen decomposition
            E1, V = torch.symeig(M1, eigenvectors = True)
            _, U = torch.symeig(M2, eigenvectors = True)

            D = torch.sqrt(torch.clamp(E1, 1e-7, 1.))
            self.corr = torch.mean(D[-self.dim:])
            self.corr_vals = D
            U, V = torch.t(U)[-self.dim:, :], torch.t(V)[-self.dim:, :]

            D = torch.diag(D)
            A = K11 @ torch.t(V)
            B = K22 @ torch.t(U)
            s = torch.sign(torch.diag(V @ S12 @ torch.t(U)))
            B *= s

            self.A, self.B = A, B

            self.corr = torch.diag(D)
            H1_proj, H2_proj = torch.t(H1) @ A, torch.t(H2) @ B

            return torch.mean(self.corr), (H1_proj, H2_proj) # TODO: revers order!

        else:

            H1 -= self.mean_x[None, :]
            H2 -= self.mean_y[None, :]
            return (H1 @ self.A), (H2 @ self.B)


if __name__ == '__main__':

    np.random.seed(1)
    original_dim = 1000
    dim = 2
    X = np.random.rand(1000, original_dim) - 0.5
    Y = np.random.rand(1000, original_dim) - 0.5

    mixing_matrix = np.random.rand(original_dim, original_dim) - 0.5
    #Y = 1 * X.copy() + 1 * (X.copy()[:, ::-1])
    Y = X.copy().dot(mixing_matrix)
    #X, Y = np.concatenate([X, Y]), np.concatenate([Y, X])
    print(X.shape, Y.shape)


    print("-----------------------------------------------------------------")
    print("gold svd")
    model = CCA(n_components=dim, tol=1e-8, max_iter=50000)
    start = time.time()
    model.fit(X.copy(), Y.copy())
    print(time.time() - start)
    x, y = model.transform(X, Y)
    print(np.real(x[:10]))
    print()
    print(np.real(y[:10]))



    print("-------------------------------------------------------------------")
    print("Full svd, training")
    model = CCAModel(dim=dim)
    start = time.time()
    x, y, corr = model(H1=X.copy(), H2=Y.copy(), training=True, alternative=True)
    print(time.time() - start)
    print(np.real(x[:10]))
    print()
    print(np.real(y[:10]))
    print(corr)
    main_x, main_y = x[:100, -1], y[:100,-1]
    print((np.sign(main_x) == np.sign(main_y)))
    print(list(zip(main_x, main_y))[:10])


    print("-------------------------------------------------------------------")
    print("Full svd, inference")
    start = time.time()
    x,y = model(H1=X.copy(), H2=Y.copy(), training=False)
    main_x, main_y = x[:100, -1], y[:100,-1]
    print((np.sign(main_x) == np.sign(main_y)))
    print(list(zip(main_x, main_y))[:10])

    print("-------------------------------------------------------------------")
    print("CCA LAYER")
    H1, H2 = torch.from_numpy(X).float().cuda(), torch.from_numpy(Y).float().cuda()
    model = CCALayer(dim = dim)
    corr, (x, y) = model(H1=H1, H2=H2)

