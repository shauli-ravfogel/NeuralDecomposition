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
            print(H1.shape, self.mean_x.shape)
            H1 -= self.mean_x[None, :]
            x_proj = (H1.dot(self.A))[:, ::-1]

            if H2 is not None:
                H2 -= self.mean_y[None, :]
                y_proj = (H2.dot(self.B))[:, ::-1]
                return x_proj, y_proj

            return x_proj
