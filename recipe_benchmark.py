import time
from scipy.sparse import issparse, random
import numpy as np
import scipy.sparse

import dask.array as da
import dask.array.random
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

def time_numpy():
    print("time_numpy")

    t0 = time.time()
    X = np.random.rand(100000, 3000)
    t1 = time.time()
    print("time to create matrix: ", t1-t0)

    Y, number_per_gene = filter_genes(X, 50000)
    Y = log1p(Y)
    Y = scale(Y)
    Y = Y.sum() # call sum so we don't have to allocate output
    t2 = time.time()
    print("time to call filter_genes: ", t2-t1)

def time_dask():
    print("time_dask")

    t0 = time.time()
    #X = dask.array.random.random((100000, 3000), chunks=(10000, 3000))
    X = np.random.rand(100000, 3000)
    X = dask.array.from_array(X, chunks=(10000, 3000))
    t1 = time.time()
    print("time to create matrix: ", t1-t0)

    Y, number_per_gene = filter_genes(X, 50000)
    Y = log1p(Y)
    Y = scale(Y)
    Y = Y.sum() # call sum so we don't have to allocate output
    da.compute(Y, number_per_gene)
    t2 = time.time()
    print("time to call filter_genes: ", t2-t1)

def time_sparse():
    print("time_sparse")

    t0 = time.time()
    X = random(100000, 3000, 0.10, format='csr') # gene expression matrix from 10x is 7% sparse, so this is similar
    #X = scipy.sparse.vstack([X] * 10)
    t1 = time.time()
    print("time to create matrix: ", t1-t0)

    Y, number_per_gene = filter_genes(X, 5000, sparse=True)
    Y = log1p(X)
    Y = densify(Y)
    Y = scale(Y)
    t2 = time.time()
    print("time to call filter_genes: ", t2-t1)

def time_sparse_dask():
    print("time_sparse_dask")

    t0 = time.time()
    X = random(100000, 3000, 0.10, format='csr') # gene expression matrix from 10x is 7% sparse, so this is similar
    #X = scipy.sparse.vstack([X] * 10)
    X = sparse_dask(X, chunks=(10000, X.shape[1]))
    t1 = time.time()
    print("time to create matrix: ", t1-t0)

    Y, number_per_gene = filter_genes(X, 5000)
    Y = log1p(Y)
    Y = Y.map_blocks(densify)
    Y = scale(Y)
    da.compute(Y, number_per_gene)
    t2 = time.time()
    print("time to call filter_genes: ", t2-t1)

def time_pydata_sparse_dask():
    print("time_pydata_sparse_dask")

    import sparse

    t0 = time.time()
    X = dask.array.random.random((100000, 3000), chunks=(10000, 3000))
    X[X < 0.90] = 0
    X = X.map_blocks(sparse.COO)
    X.compute()
    #X = sparse.random((100000, 3000), 0.10) # gene expression matrix from 10x is 7% sparse, so this is similar
    #X = da.from_array(X, chunks=(10000, X.shape[1]))
    t1 = time.time()
    print("time to create matrix: ", t1-t0)

    Y, number_per_gene = filter_genes(X, 5000)
    da.compute(Y, number_per_gene)
    t2 = time.time()
    print("time to call filter_genes: ", t2-t1)

if __name__ == '__main__':
    # Comment out to see how long tasks take
    #from dask.distributed import Client
    #client = Client(processes=False)

    # Compare dense matrices
    # time_numpy: 5.4s to create matrix, 10s to run recipe
    # time_dask: 8.4s to create matrix, 4.6s to run recipe
    # So we see that Dask can take advantage of cores to do the processing faster.

    #time_numpy()

    #time_dask()

    # Compare sparse matrices. We start with sparse matrices, but then convert to
    # dense before scaling, since this causes the matrix to become dense anyway.
    # We have scipy.sparse, Dask with a wrapper around scipy.sparse, and Dask with pydata sparse.
    # The latter doesn't currently work.

    # time_sparse: 30s to create matrix, 10s to run recipe
    # time_sparse_dask: 29s to create matrix, 3.3s to run recipe
    # So we see that Dask again can take advantage of cores.

    time_sparse()
    time_sparse_dask()
    #time_pydata_sparse_dask()

