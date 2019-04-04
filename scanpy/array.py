import numbers
import numpy as np
import scipy.sparse
from scipy.sparse import issparse

def sparse_dask(arr, chunks):
    return SparseArray(arr).asdask(chunks)

def _calculation_method(name):
    def calc(self, axis, out=None, dtype=None, **kwargs):
        if axis == 0 or axis == 1:
            return getattr(self.value, name)(axis).A.squeeze()
        elif isinstance(axis, tuple) and len(axis) == 1 and (axis[0] == 0 or axis[0] == 1):
            return getattr(self.value, name)(axis[0]).A
        elif isinstance(axis, tuple):
            v = self.value
            for ax in axis:
                v = getattr(v, name)(ax)
            return SparseArray(scipy.sparse.csr_matrix(v))
        return SparseArray(scipy.sparse.csr_matrix(getattr(self.value, name)(axis)))
    return calc

class SparseArray(np.lib.mixins.NDArrayOperatorsMixin):
    """
    An wrapper around scipy.sparse to allow sparse arrays to be the chunks in a dask array.
    """

    __array_priority__ = 10.0

    def __init__(self, value):
        self.value = value

    def __array__(self, dtype=None, **kwargs):
        # respond to np.asarray
        if isinstance(self.value, np.ndarray):
            x = self.value
        else:
            x = self.value.toarray()
        if dtype and x.dtype != dtype:
            x = x.astype(dtype)
        return x

    # One might also consider adding the built-in list type to this
    # list, to support operations like np.add(array_like, list)
    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get('out', ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use SparseArray instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle SparseArray objects.
            if not isinstance(x, self._HANDLED_TYPES + (SparseArray,)):
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(x.value if isinstance(x, SparseArray) else x
                       for x in inputs)
        if out:
            kwargs['out'] = tuple(
                x.value if isinstance(x, SparseArray) else x
                for x in out)
        # special case multiplication for sparse input, so it is elementwise, not matrix multiplication
        if ufunc.__name__ == 'multiply' and len(inputs) == 2 and issparse(inputs[0]):
            result = inputs[0].multiply(inputs[1])
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
            return self.value.__getitem__(item).toarray().squeeze()
        elif isinstance(item, tuple) and (isinstance(item[0], numbers.Number) or isinstance(item[1], numbers.Number)):
            return self.value.__getitem__(item).toarray().squeeze()
        return SparseArray(self.value.__getitem__(item))

    def __lt__(self, other):
        if isinstance(other, SparseArray):
            v = other.value
        else:
            v = other
        return SparseArray(self.value < v)
    def __le__(self, other):
        if isinstance(other, SparseArray):
            v = other.value
        else:
            v = other
        return SparseArray(self.value <= v)
    def __eq__(self, other):
        if isinstance(other, SparseArray):
            v = other.value
        else:
            v = other
        return SparseArray(self.value == v)
    def __ne__(self, other):
        if isinstance(other, SparseArray):
            v = other.value
        else:
            v = other
        return SparseArray(self.value != v)
    def __gt__(self, other):
        if isinstance(other, SparseArray):
            v = other.value
        else:
            v = other
        return SparseArray(self.value > v)
    def __ge__(self, other):
        if isinstance(other, SparseArray):
            v = other.value
        else:
            v = other
        return SparseArray(self.value >= v)

    def astype(self, dtype, copy=True):
        dtype = dtype if isinstance(dtype, np.dtype) else np.dtype(dtype)
        if copy:
            return SparseArray(self.value.astype(dtype, copy=copy))
        else:
            self.value = self.value.astype(dtype, copy=copy)
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

# TODO: make following conditional on dask being installed

from dask.array.core import concatenate_lookup

def _concatenate(L, axis=0):
    if axis == 0:
        return SparseArray(scipy.sparse.vstack(tuple([sa.value for sa in L])))
    elif axis == 1:
        return SparseArray(scipy.sparse.hstack(tuple([sa.value for sa in L])))
    else:
        msg = ("Can only concatenate sparse matrices for axis in "
               "{0, 1}.  Got %s" % axis)
        raise ValueError(msg)
concatenate_lookup.register(SparseArray, _concatenate)
