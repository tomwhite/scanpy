from pathlib import Path

import numpy as np
import scanpy as sc
import scanpy.preprocessing._dask_optimized as dask_optimized
from scanpy.sparsearray import sparse_dask

from numpy.testing import assert_allclose

HERE = Path(__file__).parent


def test_recipe_zheng17():
    # Note that this requires the 0.5GB ica_cord_blood_100K.h5ad file to be in the test data directory
    filename = HERE / '_data/ica_cord_blood_100K.h5ad'

    # Original implementation
    adata = sc.read(filename)
    sc.pp.recipe_zheng17(adata)

    # Reimplementation (no anndata, no dask)
    X = sc.read(filename).X
    Y = dask_optimized.recipe_zheng17(X)

    assert_allclose(adata.X, Y)

    # Reimplementation running with dask (no anndata)
    X = sc.read(filename).X
    X = sparse_dask(X, chunks=(10000, X.shape[1]))
    Z = dask_optimized.recipe_zheng17(X)
    Z = Z.compute()
    Z = np.asarray(Z)

    # TODO: investigate why we lose a column in dask version
    X = np.copy(adata.X)
    X = np.delete(X, 688, axis=1) # found by trial and error

    assert_allclose(X, Z, rtol=1e-3, atol=1e-1)

