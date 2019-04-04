from ._scipy_sparse import SparseArray, sparse_dask, row_scale

try:
    import cupyx
    from ._cupy_sparse import CupySparseArray, cupy_sparse_dask
except ImportError:
    pass
