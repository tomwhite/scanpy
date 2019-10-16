from pathlib import Path

import scanpy as sc
import scanpy.preprocessing._dask_optimized as dask_optimized

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
    Y = dask_optimized.recipe_zheng17(X)
