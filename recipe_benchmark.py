import time
from scipy.sparse import issparse, random
import numpy as np

import dask.array as da
import dask.array.random
import scanpy as sc
from scanpy.sparsearray import sparse_dask
from scanpy.preprocessing._dask_optimized import filter_genes, filter_genes_dispersion, normalize, log1p, densify, scale, recipe_zheng17

np.random.seed(42)

def load_data():
    return sc.read('ica_cord_blood_100K.h5ad')
    #return sc.read_10x_h5('1M_neurons_filtered_gene_bc_matrices_h5.h5')

def report_block_info(X, block_info=None):
    print(block_info, type(X))
    return X

def time_numpy():
    print("time_numpy")

    t0 = time.time()
    X = np.random.rand(100000, 3000)
    t1 = time.time()
    print("time to create matrix: ", t1-t0)

    Y, number_per_gene = filter_genes(X, X.shape[0] / 2)
    Y = normalize(Y)
    Y = log1p(Y)
    Y = scale(Y)
    Y = Y.sum() # call sum so we don't have to allocate output
    t2 = time.time()
    print("time to run recipe: ", t2-t1)

def time_dask():
    print("time_dask")

    t0 = time.time()
    #X = dask.array.random.random((100000, 3000), chunks=(10000, 3000))
    X = np.random.rand(100000, 3000)
    X = dask.array.from_array(X, chunks=(10000, 3000))
    t1 = time.time()
    print("time to create matrix: ", t1-t0)

    Y, number_per_gene = filter_genes(X, X.shape[0] / 2)
    Y = normalize(Y)
    Y = log1p(Y)
    Y = scale(Y)
    Y = Y.sum() # call sum so we don't have to allocate output
    da.compute(Y, number_per_gene)
    t2 = time.time()
    print("time to run recipe: ", t2-t1)

def time_dask_cupy():
    print("time_dask_cupy")

    import cupy

    t0 = time.time()
    #X = dask.array.random.random((100000, 3000), chunks=(10000, 3000))
    X = cupy.array(np.random.rand(100000, 3000))
    X = dask.array.from_array(X, chunks=(10000, 3000))
    t1 = time.time()
    print("time to create matrix: ", t1-t0)

    Y, number_per_gene = filter_genes(X, X.shape[0] / 2)
    Y = normalize(Y)
    Y = log1p(Y)
    Y = scale(Y)
    Y = Y.sum() # call sum so we don't have to allocate output
    da.compute(Y, number_per_gene)
    t2 = time.time()
    print("time to run recipe: ", t2-t1)

def time_sparse():
    print("time_sparse")

    t0 = time.time()
    X = random(100000, 3000, 0.10, format='csr') # gene expression matrix from 10x is 7% sparse, so this is similar
    #X = scipy.sparse.vstack([X] * 10)
    t1 = time.time()
    print("time to create matrix: ", t1-t0)

    Y, number_per_gene = filter_genes(X, X.shape[0] * 0.10 / 2)
    Y = normalize(Y)
    Y = log1p(Y)
    Y = densify(Y)
    Y = scale(Y)
    t2 = time.time()
    print("time to run recipe: ", t2-t1)

def time_sparse_dask():
    print("time_sparse_dask")

    t0 = time.time()
    X = random(100000, 3000, 0.10, format='csr') # gene expression matrix from 10x is 7% sparse, so this is similar
    #X = scipy.sparse.vstack([X] * 10)
    X = sparse_dask(X, chunks=(10000, X.shape[1]))
    t1 = time.time()
    print("time to create matrix: ", t1-t0)

    Y, number_per_gene = filter_genes(X, X.shape[0] * 0.10 / 2)
    Y = normalize(Y)
    Y = log1p(Y)
    Y = densify(Y)
    Y = scale(Y)
    da.compute(Y, number_per_gene)
    #Y.visualize(filename='sparse_dask.svg')
    t2 = time.time()
    print("time to run recipe: ", t2-t1)

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

    Y, number_per_gene = filter_genes(X, X.shape[0] * 0.10 / 2)
    da.compute(Y, number_per_gene)
    t2 = time.time()
    print("time to run recipe: ", t2-t1)

def time_sparse_original_real():
    print("time_sparse_original_real")

    t0 = time.time()
    adata = load_data()
    t1 = time.time()
    print("time to create matrix: ", t1-t0)

    sc.pp.recipe_zheng17(adata)
    t2 = time.time()
    print("time to run recipe: ", t2-t1)

def time_sparse_real():
    print("time_sparse_real")

    t0 = time.time()
    X = load_data().X
    t1 = time.time()
    print("time to create matrix: ", t1-t0)

    Y = recipe_zheng17(X)
    t2 = time.time()
    print("time to run recipe: ", t2-t1)

def time_sparse_dask_real():
    print("time_sparse_dask_real")

    t0 = time.time()
    X = load_data().X
    X = sparse_dask(X, chunks=(10000, X.shape[1]))
    t1 = time.time()
    print("time to create matrix: ", t1-t0)

    Y = recipe_zheng17(X)
    da.compute(Y)
    #Y.visualize(filename='sparse_dask.svg')
    t2 = time.time()
    print("time to run recipe: ", t2-t1)

def sparse_comparison():
    print("sparse_comparison")

    # Reimplementation (no scanpy, anndata)
    adata = load_data()
    Y, number_per_gene = filter_genes(adata.X, 1)
    Y = normalize(Y)
    Y = filter_genes_dispersion(Y, n_top_genes=1000)
    Y = normalize(Y)
    Y = log1p(Y)
    Y = densify(Y)
    Y = scale(Y)

    # Reimplementation in dask (no scanpy, anndata)
    # TODO: get this to match
    # adata = load_data()
    # X = sparse_dask(adata.X, chunks=(10000, adata.X.shape[1]))
    # Y, number_per_gene = filter_genes(X, 1)
    # Y = normalize(Y)
    # Y = filter_genes_dispersion(Y, n_top_genes=1000)
    # Y = normalize(Y)
    # Y = log1p(Y)
    # Y = densify(Y)
    # Y = scale(Y)

    # Scanpy, anndata
    adata = load_data()
    # sc.pp.filter_genes(adata, min_counts=1)
    # sc.pp.normalize_total(adata,  # normalize with total UMI count per cell
    #                       key_added='n_counts_all')
    # filter_result = sc.pp.filter_genes_dispersion(adata.X, flavor='cell_ranger', n_top_genes=1000, log=False)
    # adata._inplace_subset_var(filter_result.gene_subset)  # filter genes
    # sc.pp.normalize_total(adata)
    # sc.pp.log1p(adata)
    # sc.pp.scale(adata)

    # Use the recipe (the above is commented out but can be useful for debugging)
    sc.pp.recipe_zheng17(adata)

    # Are they the same?
    #print((adata.X!=Y).nnz) # sparse
    print(np.allclose(adata.X, Y)) # dense

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
    #time_dask_cupy()

    # Compare sparse matrices. We start with sparse matrices, but then convert to
    # dense before scaling, since this causes the matrix to become dense anyway.
    # We have scipy.sparse, Dask with a wrapper around scipy.sparse, and Dask with pydata sparse.
    # The latter doesn't currently work.

    # time_sparse: 30s to create matrix, 5.1s to run recipe
    # time_sparse_dask: 29s to create matrix, 4.3s to run recipe
    # So we see that Dask again can take advantage of cores (but not as much?)

    #time_sparse()
    #time_sparse_dask()
    #time_pydata_sparse_dask()

    # Use real data.
    # On a 64 core machine, time_sparse_real 5.3s, time_sparse_dask_real 3.0s
    # Using the 1M cell dataset: time_sparse_real 334s, time_sparse_dask_real 138s, a 2.4x speedup
    time_sparse_original_real()
    time_sparse_real()
    time_sparse_dask_real()

    #sparse_comparison()
