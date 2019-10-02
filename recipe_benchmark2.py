import time
from scipy.sparse import issparse, random
import numpy as np
import scipy.sparse
from sklearn.utils import sparsefuncs

import dask.array as da
import dask.array.random
import scanpy as sc
from scanpy.array import sparse_dask, SparseArray

np.random.seed(42)

def filter_genes(X, min_number, sparse=False):
    number_per_gene = np.sum(X, axis=0)
    if sparse:
        number_per_gene = number_per_gene.A1
    gene_subset = number_per_gene >= min_number
    s = np.sum(~gene_subset)
    print("Filtered out", s)
    print("gene_subset is a ", type(gene_subset))
    Y = X[:,gene_subset]
    return Y, number_per_gene # note we are returning "side data"

def normalize(X, sparse=False):
    counts_per_cell = X.sum(1)
    counts = np.ravel(counts_per_cell)
    after = np.median(counts[counts>0])
    counts += (counts == 0)
    counts /= after
    if sparse:
        sparsefuncs.inplace_row_scale(X, 1/counts)
    else:
        X /= counts[:, None]
    return X

def log1p(X):
    # TODO: try using out=X
    return np.log1p(X)

def densify(X):
    if issparse(X):
        return X.toarray()
    return np.asarray(X)

def scale(X):
    mean, var = _get_mean_var(X)
    scale = np.sqrt(var)
    X -= mean
    scale[scale == 0] = 1e-12
    X /= scale
    return X

def _get_mean_var(X):
    mean = X.mean(axis=0)
    mean_sq = np.multiply(X, X).mean(axis=0)
    var = (mean_sq - mean ** 2) * (X.shape[0] / (X.shape[0] - 1))
    return mean, var

def load_data():
    return sc.read('ica_cord_blood_100K.h5ad')

def sparse_comparison():
    print("sparse_comparison")

    adata = load_data()

    print(adata.X.shape)

    sc.pp.filter_genes(adata, min_counts=1)

    print("Hi 1")
    print(adata.X.shape)

    Y, _ = filter_genes(adata.X, 1, sparse=True)

    print("Hi 2")
    print(Y.shape)

    # Are they the same?
    print((adata.X!=Y).nnz)

    #from numpy.testing import assert_allclose
    #assert_allclose(adata.X.toarray(), Y.toarray())

if __name__ == '__main__':
    sparse_comparison()

