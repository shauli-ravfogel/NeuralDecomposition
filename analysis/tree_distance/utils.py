import numpy as np
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity


def triu(x: np.ndarray) -> np.ndarray:
    "Extracts upper triangular part of a matrix, excluding the diagonal."
    ones = np.ones_like(x)
    return x[np.triu(ones, k=1) == 1]


def pearson_r(x, y, axis=0, eps=1e-8):
    """Returns Pearson's correlation coefficient.
     When eps is a small number, this function will return a non-nan value
     even if x or y is constant.
     """
    from numpy.linalg import norm
    x1 = x - x.mean(axis=axis)
    x2 = y - y.mean(axis=axis)
    w12 = np.sum(x1 * x2, axis=axis)
    w1 = norm(x1, 2, axis=axis)
    w2 = norm(x2, 2, axis=axis)
    return w12 / np.maximum(eps, (w1 * w2))


def pairwise(f, data1, data2=None, parallel=False, **kwargs):
    if parallel:
        return _pairwise_parallel(f, data1, data2, **kwargs)
    else:
        return _pairwise(f, data1, data2, **kwargs)


def _pairwise(f, data1, data2=None, normalize=False, dtype=np.float64):
    """Compute matrix of values of function f applied to elements of data1 and data2."""
    symmetric = False
    if data2 is None:
        data2 = data1
        symmetric = True
    M = np.zeros((len(data1), len(data2)), dtype=dtype)
    if normalize:
        self1 = np.array([f(d, d) for d in data1], dtype=dtype)
        self2 = self1 if symmetric else np.array([f(d, d) for d in data2], dtype=dtype)
    for i, d1 in enumerate(data1):
        print("Completed row {}".format(i))
        for j, d2 in enumerate(data2):
            denom = (self1[i] * self2[j]) ** 0.5 if normalize else 1.0
            if symmetric and i > j:  # No need to re-compute lower triangular
                M[i, j] = M[j, i]
            else:
                M[i, j] = f(d1, d2) / denom
    return M


def _pairwise_parallel(f, data1, data2=None, normalize=False, backend="loky", n_jobs=-1, dtype=np.float64):
    """Compute matrix of values of function f applied to elements of data1 and data2."""
    from joblib import Parallel, delayed
    symmetric = False
    if data2 is None:
        data2 = data1
        symmetric = True
    if normalize:
        self1 = np.array([f(d, d) for d in data1], dtype=dtype)
        self2 = self1 if symmetric else np.array([f(d, d) for d in data2], dtype=dtype)
    else:
        self1 = None
        self2 = None
    M = np.array(
        Parallel(n_jobs=n_jobs, backend=backend)(delayed(compute_value)(f, self1, self2, i, j, d1, d2, normalize)
                                                 for i, d1 in enumerate(data1) for j, d2 in enumerate(data2)),
        dtype=dtype).reshape((len(data1), len(data2)))
    return M


def compute_value(f, self1, self2, i, j, d1, d2, normalize):
    denom = (self1[i] * self2[j]) ** 0.5 if normalize else 1.0
    return f(d1, d2) / denom
