import pytest
import numpy as np
from numpy.testing import assert_allclose
import scipy.sparse

from scanpy.array import SparseArray

class TestSparseArray:
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
    def xs(self, x):
        return SparseArray(scipy.sparse.csr_matrix(x))

    def test_identity(self, x, xs):
        assert_allclose(np.asarray(xs), x)

    def test_astype(self, x, xs):
        xs = xs.astype(int)
        x = x.astype(int)
        assert xs.dtype == x.dtype
        assert_allclose(np.asarray(xs), x)

    def test_astype_inplace(self, x, xs):
        original_id = id(xs)
        xs = xs.astype(int, copy=False)
        assert original_id == id(xs)
        x = x.astype(int, copy=False)
        assert xs.dtype == x.dtype
        assert_allclose(np.asarray(xs), x)

    def test_asarray(self, x, xs):
        assert_allclose(np.asarray(xs), x)

    # Not supported for sparse?
    # def test_scalar_arithmetic(self, x, xs):
    #     xs = (((xs + 1) * 2) - 4) / 1.1
    #     x = (((x + 1) * 2) - 4) / 1.1
    #     assert_allclose(np.asarray(xs), x)

    def test_arithmetic(self, x, xs):
        xs = xs * 2 + xs
        x = x * 2 + x
        assert_allclose(np.asarray(xs), x)

    # def test_broadcast_row(self, x, xs):
    #     a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    #     xs = xs + a
    #     x = x + a
    #     assert_allclose(np.asarray(xs), x)

    # def test_broadcast_col(self, x, xs):
    #     a = np.array([[1.0], [2.0], [3.0]])
    #     xs = xs + a
    #     x = x + a
    #     assert_allclose(np.asarray(xs), x)

    # TODO: implement __eq__ properly?
    # def test_eq(self, x, xs):
    #     xs = xs == 0.0
    #     x = x == 0.0
    #     assert xs.dtype == x.dtype
    #     assert_allclose(np.asarray(xs), x)

    # def test_ne(self, x, xs):
    #     xs = xs != 0.0
    #     x = x != 0.0
    #     assert_allclose(np.asarray(xs), x)

    # def test_invert(self, x, xs):
    #     xs = ~(xs == 0.0)
    #     x = ~(x == 0.0)
    #     assert_allclose(np.asarray(xs), x)

    # def test_inplace(self, x, xs):
    #     original_id = id(xs)
    #     xs += 1
    #     assert original_id == id(xs)
    #     x += 1
    #     assert_allclose(np.asarray(xs), x)

    def test_simple_index(self, x, xs):
        xs = xs[0]
        x = x[0]
        assert_allclose(xs, x)

    def test_boolean_index(self, x, xs):
        xs = np.sum(xs, axis=1)  # sum rows
        xs = xs[xs > 5]
        x = np.sum(x, axis=1)  # sum rows
        x = x[x > 5]
        assert_allclose(np.asarray(xs), x)

    def test_slice_cols(self, x, xs):
        xs = xs[:, 1:3]
        x = x[:, 1:3]
        assert xs.shape == x.shape
        assert_allclose(np.asarray(xs), x)

    def test_slice_rows(self, x, xs):
        xs = xs[1:3, :]
        x = x[1:3, :]
        assert xs.shape == x.shape
        assert_allclose(np.asarray(xs), x)

    def test_subset_cols_boolean(self, x, xs):
        subset = np.array([True, False, True, False, True])
        xs = xs[:, subset]
        x = x[:, subset]
        assert xs.shape == x.shape
        assert_allclose(np.asarray(xs), x)

    def test_subset_rows_boolean(self, x, xs):
        subset = np.array([True, False, True])
        xs = xs[subset, :]
        x = x[subset, :]
        assert xs.shape == x.shape
        assert_allclose(np.asarray(xs), x)

    def test_subset_cols_int(self, x, xs):
        subset = np.array([1, 3])
        xs = xs[:, subset]
        x = x[:, subset]
        assert xs.shape == x.shape
        assert_allclose(np.asarray(xs), x)

    def test_subset_rows_int(self, x, xs):
        subset = np.array([1, 2])
        xs = xs[subset, :]
        x = x[subset, :]
        assert xs.shape == x.shape
        assert_allclose(np.asarray(xs), x)

    # def test_newaxis(self, x, xs):
    #     xs = np.sum(xs, axis=1)[:, np.newaxis]
    #     x = np.sum(x, axis=1)[:, np.newaxis]
    #     assert_allclose(np.asarray(xs), x)

    def test_log1p(self, x, xs):
        log1pnps = np.asarray(np.log1p(xs))
        log1pnp = np.log1p(x)
        assert_allclose(log1pnps, log1pnp)

    def test_sum(self, x, xs):
        totald = np.sum(xs)
        total = np.sum(x)
        assert totald == pytest.approx(total)

    def test_sum_cols(self, x, xs):
        xs = np.sum(xs, axis=0)
        x = np.sum(x, axis=0)
        assert_allclose(np.asarray(xs), x)

    def test_sum_rows(self, x, xs):
        xs = np.sum(xs, axis=1)
        x = np.sum(x, axis=1)
        assert_allclose(np.asarray(xs), x)

    def test_mean(self, x, xs):
        meand = np.mean(xs)
        mean = np.mean(x)
        assert meand == pytest.approx(mean)

    def test_mean_cols(self, x, xs):
        xs = np.mean(xs, axis=0)
        x = np.mean(x, axis=0)
        assert_allclose(np.asarray(xs), x)

    def test_mean_rows(self, x, xs):
        xs = np.mean(xs, axis=1)
        x = np.mean(x, axis=1)
        assert_allclose(np.asarray(xs), x)

    def test_multiply(self, x, xs):
        xs = np.multiply(xs, xs) # elementwise, not matrix multiplication
        x = np.multiply(x, x)
        assert_allclose(np.asarray(xs), x)

    def test_var(self, x, xs):
        def var(x):
            mean = x.mean(axis=0)
            mean_sq = np.multiply(x, x).mean(axis=0)
            return mean_sq - mean ** 2

        varnps = np.asarray(var(xs))
        varnp = var(x)
        assert_allclose(varnps, varnp)

    def test_median(self, x, xs):
        mediand = np.median(xs)
        median = np.median(x)
        assert mediand == pytest.approx(median)
