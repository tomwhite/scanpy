import time
from scipy.sparse import random
import numpy as np
import scipy.sparse

import dask.array as da
import dask.array.random
from scanpy.array import sparse_dask

np.random.seed(42)

def filter_genes(X):
    min_number = 50000
    number_per_gene = np.sum(X, axis=0)
    gene_subset = number_per_gene >= min_number
    #s = np.sum(~gene_subset)
    Y = X[:,gene_subset]
    return Y, number_per_gene # note we are returning "side data"

def log1p(X):
    # TODO: try using out=X
    return np.log1p(X)

def time_numpy():
    print("time_numpy")

    t0 = time.time()
    X = np.random.rand(100000, 3000)
    t1 = time.time()
    print("time to create matrix: ", t1-t0)

    Y, number_per_gene = filter_genes(X)
    Y = log1p(Y).sum() # call sum so we don't have to allocate output
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

    Y, number_per_gene = filter_genes(X)
    Y = log1p(Y).sum() # call sum so we don't have to allocate output
    da.compute(Y, number_per_gene)
    t2 = time.time()
    print("time to call filter_genes: ", t2-t1)

if __name__ == '__main__':
    time_numpy()

    # For seeing how long tasks take
    #from dask.distributed import Client
    #client = Client(processes=False)

    time_dask()

