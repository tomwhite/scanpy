import pytest
import numpy as np
from numpy.testing import assert_allclose
import scipy.sparse

from scanpy.array import sparse_dask, row_scale

# import cupy if installed
try:
    import cupyx
    import cupy as cp
except ImportError:
    cp = None

class TestCuSparseDaskArray:
    @pytest.fixture()
    def x(self):
        return np.array(
            [
                [0.0, 1.0, 0.0, 3.0, 0.0],
                [2.0, 0.0, 3.0, 4.0, 5.0],
                [4.0, 0.0, 0.0, 6.0, 7.0],
            ]
        )

    @pytest.fixture()
    def xd(self, x):
        return sparse_dask(cupyx.scipy.sparse.csr_matrix(scipy.sparse.csr_matrix(x)), chunks=(2, 5))

    def test_identity(self, x, xd):
        assert_allclose(np.asarray(xd), x)

    def test_astype(self, x, xd):
        xd = xd.astype(np.float32)
        x = x.astype(np.float32)
        assert xd.dtype == x.dtype
        assert_allclose(np.asarray(xd), x)

    # Not sure of what expected Dask behaviour is
    # def test_astype_inplace(self, x, xd):
    #     original_id = id(xd)
    #     xd = xd.astype(int, copy=False)
    #     assert original_id == id(xd)
    #     x = x.astype(int, copy=False)
    #     assert xd.dtype == x.dtype
    #     assert_allclose(np.asarray(xd), x)

    def test_asarray(self, x, xd):
        assert_allclose(np.asarray(xd), x)

    # Not supported for sparse?
    # def test_scalar_arithmetic(self, x, xd):
    #     xd = (((xd + 1) * 2) - 4) / 1.1
    #     x = (((x + 1) * 2) - 4) / 1.1
    #     assert_allclose(np.asarray(xd), x)

    def test_arithmetic(self, x, xd):
        xd = xd * 2 + xd
        x = x * 2 + x
        assert_allclose(np.asarray(xd), x)

    # def test_broadcast_row(self, x, xd):
    #     a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    #     xd = xd + a
    #     x = x + a
    #     assert_allclose(np.asarray(xd), x)

    # def test_broadcast_col(self, x, xd):
    #     a = np.array([[1.0], [2.0], [3.0]])
    #     xd = xd + a
    #     x = x + a
    #     assert_allclose(np.asarray(xd), x)

    # def test_broadcast_col(self, x, xd):
    #     a = np.array([[1.0], [2.0], [3.0]])
    #     xd = xd / a
    #     x = x / a
    #     assert_allclose(np.asarray(xd), x)

    def test_row_scale(self, x, xd):
        a = np.array([[1.0], [2.0], [3.0]])
        xd = row_scale(xd, 1 / a)
        x = x / a
        assert_allclose(np.asarray(xd), x)

    # TODO: implement __eq__ properly?
    # def test_eq(self, x, xd):
    #     xd = xd == 0.0
    #     x = x == 0.0
    #     assert xd.dtype == x.dtype
    #     assert_allclose(np.asarray(xd), x)

    # def test_ne(self, x, xd):
    #     xd = xd != 0.0
    #     x = x != 0.0
    #     assert_allclose(np.asarray(xd), x)

    # def test_invert(self, x, xd):
    #     xd = ~(xd == 0.0)
    #     x = ~(x == 0.0)
    #     assert_allclose(np.asarray(xd), x)

    # def test_inplace(self, x, xd):
    #     original_id = id(xd)
    #     xd += 1
    #     assert original_id == id(xd)
    #     x += 1
    #     assert_allclose(np.asarray(xd), x)

    def test_simple_index(self, x, xd):
        xd = xd[0]
        x = x[0]
        assert_allclose(xd, x)

    def test_boolean_index(self, x, xd):
        xd = np.sum(xd, axis=1)  # sum rows
        xd = xd[xd > 5]
        x = np.sum(x, axis=1)  # sum rows
        x = x[x > 5]
        assert_allclose(np.asarray(xd), x)

    def test_boolean_index_simplified(self, x, xd):
        xd = xd > 5
        x = x > 5
        assert_allclose(np.asarray(xd), x)

    def test_slice_cols(self, x, xd):
        xd = xd[:, 1:3]
        x = x[:, 1:3]
        assert xd.shape == x.shape
        assert_allclose(np.asarray(xd), x)

    def test_slice_rows(self, x, xd):
        xd = xd[1:3, :]
        x = x[1:3, :]
        assert xd.shape == x.shape
        assert_allclose(np.asarray(xd), x)

    def test_subset_cols_boolean(self, x, xd):
        subset = np.array([True, False, True, False, True])
        xd = xd[:, subset]
        x = x[:, subset]
        assert xd.shape == x.shape
        assert_allclose(np.asarray(xd), x)

    def test_subset_rows_boolean(self, x, xd):
        subset = np.array([True, False, True])
        xd = xd[subset, :]
        x = x[subset, :]
        assert xd.shape == x.shape
        assert_allclose(np.asarray(xd), x)

    def test_subset_cols_int(self, x, xd):
        subset = np.array([1, 3])
        xd = xd[:, subset]
        x = x[:, subset]
        assert xd.shape == x.shape
        assert_allclose(np.asarray(xd), x)

    def test_subset_rows_int(self, x, xd):
        subset = np.array([1, 2])
        xd = xd[subset, :]
        x = x[subset, :]
        assert xd.shape == x.shape
        assert_allclose(np.asarray(xd), x)

    # def test_newaxis(self, x, xd):
    #     xd = np.sum(xd, axis=1)[:, np.newaxis]
    #     x = np.sum(x, axis=1)[:, np.newaxis]
    #     assert_allclose(np.asarray(xd), x)

    def test_log1p(self, x, xd):
        log1pnps = np.asarray(np.log1p(xd))
        log1pnp = np.log1p(x)
        assert_allclose(log1pnps, log1pnp)

    def test_sum(self, x, xd):
        totald = np.sum(xd)
        total = np.sum(x)
        assert totald == pytest.approx(total)

    def test_sum_cols(self, x, xd):
        xd = np.sum(xd, axis=0)
        x = np.sum(x, axis=0)
        assert_allclose(np.asarray(xd), x)

    def test_sum_rows(self, x, xd):
        xd = np.sum(xd, axis=1)
        x = np.sum(x, axis=1)
        assert_allclose(np.asarray(xd), x)

    def test_mean(self, x, xd):
        meand = np.mean(xd)
        mean = np.mean(x)
        assert meand == pytest.approx(mean)

    def test_mean_cols(self, x, xd):
        xd = np.mean(xd, axis=0)
        x = np.mean(x, axis=0)
        assert_allclose(np.asarray(xd), x)

    def test_mean_rows(self, x, xd):
        xd = np.mean(xd, axis=1)
        x = np.mean(x, axis=1)
        assert_allclose(np.asarray(xd), x)

    def test_multiply(self, x, xd):
        xd = np.multiply(xd, xd) # elementwise, not matrix multiplication
        x = np.multiply(x, x)
        assert_allclose(np.asarray(xd), x)

    def test_var(self, x, xd):
        def var(x):
            mean = x.mean(axis=0)
            mean_sq = np.multiply(x, x).mean(axis=0)
            return mean_sq - mean ** 2

        varnps = np.asarray(var(xd))
        varnp = var(x)
        assert_allclose(varnps, varnp)

    def test_median(self, x, xd):
        mediand = np.median(xd)
        median = np.median(x)
        assert mediand == pytest.approx(median)
