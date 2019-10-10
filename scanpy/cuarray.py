import numbers
import numpy as np
import cupyx
import cupy as cp
from cupyx.scipy.sparse import issparse


def sparse_dask(arr, chunks):
    return CupySparseArray(arr).asdask(chunks)

def _convert_to_numpy_array(arr, dtype=None):
    if isinstance(arr, np.ndarray):
        ret = arr
    elif cp is not None and isinstance(arr, cp.ndarray):
        ret = arr.get()
    elif cp is not None and cupyx.scipy.sparse.issparse(arr):
        ret = arr.toarray().get()
    else:
        ret = arr.toarray()
    if dtype and ret.dtype != dtype:
        ret = ret.astype(dtype)
    return ret

def _calculation_method(name):
    def calc(self, axis=None, out=None, dtype=None, **kwargs):
        if axis is None:
            # cupy sparse returns a zero dimensional array, so we call item() to get its only element
            return getattr(self.value, name)(axis).item()
        elif axis == 0 or axis == 1:
            return _convert_to_numpy_array(getattr(self.value, name)(axis)).squeeze()
        elif isinstance(axis, tuple) and len(axis) == 1 and (axis[0] == 0 or axis[0] == 1):
            return _convert_to_numpy_array(getattr(self.value, name)(axis[0]))
        elif isinstance(axis, tuple):
            v = self.value
            for ax in axis:
                v = getattr(v, name)(ax)
            return CupySparseArray(cupyx.scipy.sparse.csr_matrix(v))
        result = getattr(self.value, name)(axis)
        return CupySparseArray(cupyx.scipy.sparse.csr_matrix(result))
    return calc


class CupySparseArray(np.lib.mixins.NDArrayOperatorsMixin):
    """
    An wrapper around cupyx.scipy.sparse to allow sparse arrays to be the chunks in a dask array.
    """

    __array_priority__ = 10.0

    def __init__(self, value):
        if not issparse(value):
            raise ValueError(f"CupySparseArray only takes a cupyx.scipy.sparse value, but given {type(value)}")
        self.value = value

    def __array__(self, dtype=None, **kwargs):
        # respond to np.asarray
        return _convert_to_numpy_array(self.value, dtype)

    # One might also consider adding the built-in list type to this
    # list, to support operations like np.add(array_like, list)
    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get('out', ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use CupySparseArray instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle CupySparseArray objects.
            if not isinstance(x, self._HANDLED_TYPES + (CupySparseArray,)):
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(x.value if isinstance(x, CupySparseArray) else x
                       for x in inputs)
        if out:
            kwargs['out'] = tuple(
                x.value if isinstance(x, CupySparseArray) else x
                for x in out)
        # special case multiplication for sparse input, so it is elementwise, not matrix multiplication
        if ufunc.__name__ == 'multiply' and len(inputs) == 2 and issparse(inputs[0]) and issparse(inputs[1]):
            result = inputs[0].multiply(inputs[1])
        elif ufunc.__name__ == 'true_divide' and len(inputs) == 2 and issparse(inputs[0]) and issparse(inputs[1]):
            result = inputs[0] / inputs[1]
        else:
            result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            # multiple return values
            return tuple(type(self)(x) for x in result)
        elif method == 'at':
            # no return value
            return None
        else:
            # one return value
            return type(self)(result)

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, self.value)

    @property
    def ndim(self):
        return self.value.ndim

    @property
    def shape(self):
        return self.value.shape

    @property
    def dtype(self):
        return self.value.dtype

    def __getitem__(self, item):
        if isinstance(item, numbers.Number):
            return _convert_to_numpy_array(self.value.__getitem__(item)).squeeze()
        elif isinstance(item, tuple) and (isinstance(item[0], numbers.Number) or isinstance(item[1], numbers.Number)):
            return _convert_to_numpy_array(self.value.__getitem__(item)).squeeze()
        # replace slices that span the entire column or row with slice(None) to ensure cupy sparse doesn't blow up
        if isinstance(item[0], slice) and item[0].start == 0 and item[0].stop == self.shape[0] and item[0].step is None:
            item0 = slice(None)
        else:
            item0 = item[0]
        if isinstance(item[1], slice) and item[1].start == 0 and item[1].stop == self.shape[1] and item[1].step is None:
            item1 = slice(None)
        else:
            item1 = item[1]
        return CupySparseArray(self.value.__getitem__((item0, item1)))

    def _get_value(self, other):
        # get the value if a CupySparseArray, or just return other
        return other.value if isinstance(other, CupySparseArray) else other

    def __lt__(self, other):
        return CupySparseArray(self.value < self._get_value(other))
    def __le__(self, other):
        return CupySparseArray(self.value <= self._get_value(other))
    def __eq__(self, other):
        return CupySparseArray(self.value == self._get_value(other))
    def __ne__(self, other):
        return CupySparseArray(self.value != self._get_value(other))
    def __gt__(self, other):
        return CupySparseArray(self.value > self._get_value(other))
    def __ge__(self, other):
        return CupySparseArray(self.value >= self._get_value(other))

    def astype(self, dtype, copy=True):
        dtype = dtype if isinstance(dtype, np.dtype) else np.dtype(dtype)
        if copy:
            return CupySparseArray(self.value.astype(dtype))
        else:
            # cupy sparse doesn't support the copy argument
            self.value = self.value.astype(dtype)
            return self

    mean = _calculation_method('mean')
    argmax = _calculation_method('argmax')
    min = _calculation_method('min')
    argmin = _calculation_method('argmin')
    sum = _calculation_method('sum')
    prod = _calculation_method('prod')
    all = _calculation_method('all')
    any = _calculation_method('any')

    def asdask(self, chunks):
        import dask.array as da
        return da.from_array(self, chunks=chunks, asarray=False, fancy=False)

def _compressed_sparse_stack(blocks, axis):
    """
    Stacking fast path for CSR/CSC matrices
    (i) vstack for CSR, (ii) hstack for CSC.
    """
    other_axis = 1 if axis == 0 else 0
    data = cp.concatenate([b.data for b in blocks])
    constant_dim = blocks[0].shape[other_axis]
    # TODO: get_index_dtype doesn't work with cupy
    # from scipy.sparse.sputils import get_index_dtype
    #idx_dtype = get_index_dtype(arrays=[b.indptr for b in blocks],
    #                            maxval=max(data.size, constant_dim))
    int32max = np.iinfo(np.int32).max
    maxval = max(data.size, constant_dim)
    idx_dtype = np.int64 if maxval > int32max else np.intc
    indices = cp.empty(data.size, dtype=idx_dtype)
    indptr = cp.empty(sum(b.shape[axis] for b in blocks) + 1, dtype=idx_dtype)
    last_indptr = idx_dtype(0)
    sum_dim = 0
    sum_indices = 0
    for b in blocks:
        if b.shape[other_axis] != constant_dim:
            raise ValueError('incompatible dimensions for axis %d' % other_axis)
        indices[sum_indices:sum_indices+b.indices.size] = b.indices
        sum_indices += b.indices.size
        idxs = slice(sum_dim, sum_dim + b.shape[axis])
        indptr[idxs] = b.indptr[:-1]
        indptr[idxs] += last_indptr
        sum_dim += b.shape[axis]
        last_indptr += b.indptr[-1]
    indptr[-1] = last_indptr
    if axis == 0:
        return cupyx.scipy.sparse.csr_matrix((data, indices, indptr),
                                             shape=(sum_dim, constant_dim))
    else:
        return cupyx.scipy.sparse.csc_matrix((data, indices, indptr),
                                             shape=(constant_dim, sum_dim))

def _concatenate(L, axis=0):
    if len(L) == 1:
        return L[0]
    if axis == 0:
        return CupySparseArray(_compressed_sparse_stack(tuple([sa.value for sa in L]), 0))
    elif axis == 1:
        return CupySparseArray(_compressed_sparse_stack(tuple([sa.value for sa in L]), 1))
    else:
        msg = ("Can only concatenate sparse matrices for axis in "
               "{0, 1}.  Got %s" % axis)
        raise ValueError(msg)

# register concatenate if Dask is installed
try:
    from dask.array.core import concatenate_lookup
    concatenate_lookup.register(CupySparseArray, _concatenate)
except ImportError:
    pass
