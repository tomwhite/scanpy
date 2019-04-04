from importlib.util import find_spec
import os
from pathlib import Path

import anndata as ad
import numpy.testing as npt
import pytest
from scipy import sparse

import scanpy as sc
from scanpy.preprocessing import *
from scanpy.preprocessing._simple import materialize_as_ndarray


HERE = Path(__file__).parent / Path('_data/')
input_file = str(Path(HERE, "10x-10k-subset.zarr"))

required = ['dask', 'zappy', 'zarr']
installed = {mod: bool(find_spec(mod)) for mod in required}


@pytest.mark.skipif(
    not all(installed.values()), reason=f'{required} all required: {installed}'
)
class TestPreprocessingDistributed:
    @pytest.fixture()
    def adata(self):
        # a = ad.read_zarr(input_file)  # regular anndata
        # a.X = a.X[:]  # convert to numpy array
        a = sc.read_10x_h5(os.path.join(ROOT, '3.0.0', 'filtered_feature_bc_matrix.h5'))
        return a

    @pytest.fixture(params=["dask"])
    def adata_dist(self, request):
        a = sc.read_10x_h5(os.path.join(ROOT, '3.0.0', 'filtered_feature_bc_matrix.h5'))
        # regular anndata except for X, which we replace on the next line
        #a = ad.read_zarr(input_file)
        input_file_X = input_file + "/X"
        if request.param == "direct":
            import zappy.direct

            a.X = zappy.direct.from_zarr(input_file_X)
            yield a
        elif request.param == "dask":
            from scanpy.array import sparse_dask
            a.X = sparse_dask(a.X, chunks=(500, a.X.shape[1]))
            yield a


    def test_subsample(self, adata, adata_dist):
        subsample(adata_dist, 0.5)
        result = materialize_as_ndarray(adata_dist.X)
        subsample(adata, 0.5)
        npt.assert_allclose(result, adata.X.toarray())

    def test_log1p(self, adata, adata_dist):
        log1p(adata_dist)
        result = materialize_as_ndarray(adata_dist.X)
        log1p(adata)
        assert result.shape == adata.shape
        assert result.shape == (adata.n_obs, adata.n_vars)
        npt.assert_allclose(result, adata.X.toarray())

    def test_normalize_per_cell(self, adata, adata_dist):
        normalize_per_cell(adata_dist)
        result = materialize_as_ndarray(adata_dist.X)
        normalize_per_cell(adata)
        assert result.shape == adata.shape
        assert result.shape == (adata.n_obs, adata.n_vars)
        npt.assert_allclose(result, adata.X.toarray(), rtol=1e-05) # TODO: improve tolerance

    def test_normalize_total(self, adata, adata_dist):
        normalize_total(adata_dist)
        result = materialize_as_ndarray(adata_dist.X)
        normalize_total(adata)
        assert result.shape == adata.shape
        assert result.shape == (adata.n_obs, adata.n_vars)
        npt.assert_allclose(result, adata.X)

    def test_filter_cells(self, adata, adata_dist):
        filter_cells(adata_dist, min_genes=3)
        result = materialize_as_ndarray(adata_dist.X)
        filter_cells(adata, min_genes=3)
        assert result.shape == adata.shape
        assert result.shape == (adata.n_obs, adata.n_vars)
        npt.assert_allclose(result, adata.X.toarray())

    def test_filter_genes(self, adata, adata_dist):
        filter_genes(adata_dist, min_cells=2)
        result = materialize_as_ndarray(adata_dist.X)
        filter_genes(adata, min_cells=2)
        assert result.shape == adata.shape
        assert result.shape == (adata.n_obs, adata.n_vars)
        npt.assert_allclose(result, adata.X.toarray())

    # def test_write_zarr(self, adata, adata_dist):
    #     import dask.array as da
    #     import zarr
    #
    #     log1p(adata_dist)
    #     temp_store = zarr.TempStore()
    #     chunks = adata_dist.X.chunks
    #     # write metadata using regular anndata
    #     adata.write_zarr(temp_store, chunks)
    #     if isinstance(adata_dist.X, da.Array):
    #         adata_dist.X.to_zarr(temp_store.dir_path("X"))
    #     else:
    #         adata_dist.X.to_zarr(temp_store.dir_path("X"), chunks)
    #     # read back as zarr directly and check it is the same as adata.X
    #     adata_log1p = ad.read_zarr(temp_store)
    #     log1p(adata)
    #     npt.assert_allclose(adata_log1p.X, adata.X)
