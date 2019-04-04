import time
import numpy as np

import scanpy as sc
from scanpy.sparsearray import sparse_dask
from scanpy.preprocessing._dask_optimized import recipe_zheng17_anndata

np.random.seed(42)

def load_data():
    return sc.read('ica_cord_blood_100K.h5ad')
    #return sc.read_10x_h5('1M_neurons_filtered_gene_bc_matrices_h5.h5')

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
    adata = load_data()
    t1 = time.time()
    print("time to create matrix: ", t1-t0)

    recipe_zheng17_anndata(adata)
    t2 = time.time()
    print("time to run recipe: ", t2-t1)

def time_sparse_dask_real():
    print("time_sparse_dask_real")

    t0 = time.time()
    adata = load_data()
    adata.X = sparse_dask(adata.X, chunks=(10000, adata.X.shape[1]))
    t1 = time.time()
    print("time to create matrix: ", t1-t0)

    recipe_zheng17_anndata(adata)
    #Y.visualize(filename='sparse_dask.svg')
    t2 = time.time()
    print("time to run recipe: ", t2-t1)

if __name__ == '__main__':
    # Comment out to see how long tasks take
    #from dask.distributed import Client
    #client = Client(processes=False)

    # Use real data.
    # On a 64 core machine, time_sparse_real 5.3s, time_sparse_dask_real 3.0s
    # Using the 1M cell dataset: time_sparse_real 334s, time_sparse_dask_real 138s, a 2.4x speedup
    time_sparse_original_real()
    time_sparse_real()
    time_sparse_dask_real()
