import time
from scipy.sparse import issparse, random
import numpy as np
import scipy.sparse
from scipy.sparse import issparse
from sklearn.utils import sparsefuncs

import dask.array as da
import dask.array.random
import scanpy as sc
from scanpy.array import sparse_dask, SparseArray
import warnings

np.random.seed(42)

def materialize_as_ndarray(a):
    """Convert distributed arrays to ndarrays."""
    if type(a) in (list, tuple):
        if da is not None and any(isinstance(arr, da.Array) for arr in a):
            return da.compute(*a, sync=True)
        return tuple(np.asarray(arr) for arr in a)
    return np.asarray(a)

def filter_genes(X, min_number):
    number_per_gene = np.sum(X, axis=0)
    if issparse(X):
        number_per_gene = number_per_gene.A1
    gene_subset = number_per_gene >= min_number
    s = np.sum(~gene_subset)
    Y = X[:,gene_subset]
    return Y, number_per_gene # note we are returning "side data"

def filter_genes_dispersion(X, n_top_genes):
    # we need to materialize the mean and var since we use pandas to do computations on them
    mean, var = materialize_as_ndarray(_get_mean_var(X))
    dispersion = var / mean
    import pandas as pd
    df = pd.DataFrame()
    df['mean'] = mean
    df['dispersion'] = dispersion

    # cell ranger
    from statsmodels import robust
    df['mean_bin'] = pd.cut(df['mean'], np.r_[-np.inf,
                                              np.percentile(df['mean'], np.arange(10, 105, 5)), np.inf])
    disp_grouped = df.groupby('mean_bin')['dispersion']
    disp_median_bin = disp_grouped.median()
    # the next line raises the warning: "Mean of empty slice"
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        disp_mad_bin = disp_grouped.apply(robust.mad)
    df['dispersion_norm'] = np.abs((
        df['dispersion'].values
        - disp_median_bin[df['mean_bin'].values].values
    )) / disp_mad_bin[df['mean_bin'].values].values
    dispersion_norm = df['dispersion_norm'].values.astype('float32')

    dispersion_norm = dispersion_norm[~np.isnan(dispersion_norm)]
    dispersion_norm[::-1].sort()  # interestingly, np.argpartition is slightly slower
    disp_cut_off = dispersion_norm[n_top_genes-1]
    gene_subset = df['dispersion_norm'].values >= disp_cut_off

    return X[:,gene_subset]

def normalize(X):
    counts_per_cell = X.sum(1)
    counts = np.ravel(counts_per_cell)
    after = np.median(counts[counts>0])
    counts += (counts == 0)
    counts /= after
    if issparse(X):
        sparsefuncs.inplace_row_scale(X, 1/counts)
    else:
        X /= counts[:, None]
    return X

def normalize_sparse_dask(X):
    # Do the equivalent of inplace_row_scale for SparseArray
    counts_per_cell = X.sum(1)
    counts = np.ravel(counts_per_cell)
    after = np.median(counts[counts>0])
    counts += (counts == 0)
    counts /= after
    def inplace_row_scale(X, block_info=None):
        if block_info == '__block_info_dummy__':
            return X
        loc = block_info[0]['array-location'][0]
        return SparseArray(sparsefuncs.inplace_row_scale(X.value, 1/counts[loc[0]:loc[1]]))
    return X.map_blocks(inplace_row_scale, dtype=X.dtype)

def log1p(X):
    # TODO: try using out=X
    return np.log1p(X)

def densify(X):
    if issparse(X):
        return X.toarray()
    elif isinstance(X, da.Array):
        # densify individual blocks - useful if backed by a SparseArray
        return X.map_blocks(densify, dtype=X.dtype)
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
    if issparse(X):
        mean_sq = X.multiply(X).mean(axis=0)
        mean = mean.A1
        mean_sq = mean_sq.A1
    else:
        mean_sq = np.multiply(X, X).mean(axis=0)
    var = (mean_sq - mean ** 2) * (X.shape[0] / (X.shape[0] - 1))
    return mean, var

def load_data():
    return sc.read('ica_cord_blood_100K.h5ad')

def report_block_info(X, block_info=None):
    print(block_info, type(X))
    return X

def time_numpy():
    print("time_numpy")

    t0 = time.time()
    X = np.random.rand(100000, 3000)
    t1 = time.time()
    print("time to create matrix: ", t1-t0)

    Y, number_per_gene = filter_genes(X, 50000)
    Y = normalize(Y)
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
    Y = normalize(Y)
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

    Y, number_per_gene = filter_genes(X, 5000)
    Y = normalize(Y)
    Y = log1p(Y)
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
    Y = normalize(Y)
    Y = log1p(Y)
    Y = densify(Y)
    Y = scale(Y)
    da.compute(Y, number_per_gene)
    #Y.visualize(filename='sparse_dask.svg')
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

def time_sparse_real():
    print("time_sparse_real")

    t0 = time.time()
    X = load_data().X
    t1 = time.time()
    print("time to create matrix: ", t1-t0)

    Y, number_per_gene = filter_genes(X, 5000)
    Y = normalize(Y)
    #Y = filter_genes_dispersion(Y, n_top_genes=1000)
    #Y = normalize(Y)
    Y = log1p(Y)
    Y = densify(Y)
    Y = scale(Y)
    t2 = time.time()
    print("time to call filter_genes: ", t2-t1)

def time_sparse_dask_real():
    print("time_sparse_dask_real")

    t0 = time.time()
    X = load_data().X
    X = sparse_dask(X, chunks=(10000, X.shape[1]))
    t1 = time.time()
    print("time to create matrix: ", t1-t0)

    Y, number_per_gene = filter_genes(X, 5000)
    #Y = Y.map_blocks(report_block_info, dtype=Y.dtype)
    Y = normalize(Y)
    #Y = filter_genes_dispersion(Y, n_top_genes=1000)
    #Y = normalize(Y)
    Y = log1p(Y)
    Y = densify(Y)
    Y = scale(Y)
    da.compute(Y, number_per_gene)
    #Y.visualize(filename='sparse_dask.svg')
    t2 = time.time()
    print("time to call filter_genes: ", t2-t1)

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
    time_sparse_real()
    time_sparse_dask_real()

    #sparse_comparison()
