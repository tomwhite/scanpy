import numpy as np
from scipy.sparse import issparse
from sklearn.utils import sparsefuncs

import dask.array as da
from scanpy.sparsearray import sparse_dask, row_scale
import warnings

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
    elif isinstance(X, da.Array):
        X = row_scale(X, 1/counts)
    else:
        X /= counts[:, None]
    return X

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

def recipe_zheng17(X):
    pass
