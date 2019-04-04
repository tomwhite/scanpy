import time
from scipy.sparse import random
import numpy as np
import scipy.sparse

import dask.array.random
from scanpy.array import sparse_dask

np.random.seed(42)

#np.show_config()

def time_numpy():

    t0 = time.time()
    #X = random(100000, 3000, 0.07) # gene expression matrix from 10x is 7% sparse
    X = np.random.rand(100000, 3000)
    t1 = time.time()

    print("time to create matrix: ", t1-t0)
    #print(X.nnz/(X.shape[0] * X.shape[1]))

    #Y = np.log1p(X.data, out=X.data)
    Y = np.log1p(X, out=X)
    t2 = time.time()

    print("time to call log1p: ", t2-t1)

def time_dask():

    t0 = time.time()
    X = dask.array.random.random((100000, 3000), chunks=(10000, 3000))
    t1 = time.time()

    print("time to create matrix: ", t1-t0)

    np.log1p(X, out=X)
    X.compute(scheduler='threads', num_workers=8)
    t2 = time.time()

    print("time to call log1p: ", t2-t1)

# def random_sparse():
#     X = random(100000, 3000, 0.10, format='csr') # gene expression matrix from 10x is 7% sparse, so this is similar
#     X10 = scipy.sparse.vstack([X] * 10)
#     scipy.sparse.save_npz('/tmp/sparse_matrix.npz', X10)

def time_sparse():

    t0 = time.time()
    X = random(100000, 3000, 0.10, format='csr') # gene expression matrix from 10x is 7% sparse, so this is similar
    #X = scipy.sparse.vstack([X] * 10)
    t1 = time.time()

    print("time to create matrix: ", t1-t0)

    #print(X.nnz/(X.shape[0] * X.shape[1]))

    Y = np.log1p(X.data, out=X.data)
    t2 = time.time()

    print("time to call log1p: ", t2-t1)

def time_sparse_dask():

    t0 = time.time()
    X = random(100000, 3000, 0.10, format='csr') # gene expression matrix from 10x is 7% sparse, so this is similar
    #X = scipy.sparse.vstack([X] * 10)
    X = sparse_dask(X, chunks=(10000, X.shape[1]))
    t1 = time.time()

    print("time to create matrix: ", t1-t0)

    np.log1p(X, out=X)
    X.compute(scheduler='threads', num_workers=8)
    t2 = time.time()

    print("time to call log1p: ", t2-t1)

def time_pydata_sparse_dask():

    import sparse
    import dask.array as da

    t0 = time.time()
    X = dask.array.random.random((100000, 3000), chunks=(10000, 3000))
    X[X < 0.90] = 0
    X = X.map_blocks(sparse.COO)
    X.compute(scheduler='threads', num_workers=8)
    #X = sparse.random((100000, 3000), 0.10) # gene expression matrix from 10x is 7% sparse, so this is similar
    #X = da.from_array(X, chunks=(10000, X.shape[1]))
    t1 = time.time()

    print("time to create matrix: ", t1-t0)

    np.log1p(X, out=X)
    X.compute(scheduler='threads', num_workers=8)
    t2 = time.time()

    print("time to call log1p: ", t2-t1)

if __name__ == '__main__':
    # time_numpy()
    # time_dask()

    time_sparse()
    time_sparse_dask()
    # time_pydata_sparse_dask()
