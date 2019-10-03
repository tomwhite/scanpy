import cupy
import dask.array as da
import numpy as np
X = cupy.array(np.random.rand(10000, 100))
X = da.from_array(X, chunks=(1000, 100))
Y = X.mean(axis=0)
Y.compute()

# Traceback (most recent call last):
# File "<stdin>", line 1, in <module>
# File "/opt/anaconda3/lib/python3.7/site-packages/dask/base.py", line 175, in compute
# (result,) = compute(self, traverse=False, **kwargs)
# File "/opt/anaconda3/lib/python3.7/site-packages/dask/base.py", line 446, in compute
# results = schedule(dsk, keys, **kwargs)
# File "/opt/anaconda3/lib/python3.7/site-packages/dask/threaded.py", line 82, in get
# **kwargs
# File "/opt/anaconda3/lib/python3.7/site-packages/dask/local.py", line 491, in get_async
# raise_exception(exc, tb)
# File "/opt/anaconda3/lib/python3.7/site-packages/dask/compatibility.py", line 130, in reraise
# raise exc
# File "/opt/anaconda3/lib/python3.7/site-packages/dask/local.py", line 233, in execute_task
# result = _execute_task(task, data)
# File "/opt/anaconda3/lib/python3.7/site-packages/dask/core.py", line 119, in _execute_task
# return func(*args2)
# File "/opt/anaconda3/lib/python3.7/site-packages/dask/array/reductions.py", line 582, in mean_agg
# return divide(total, n, dtype=dtype)
# File "/opt/anaconda3/lib/python3.7/site-packages/dask/array/reductions.py", line 40, in divide
# return f(a, b, dtype=dtype)
# File "/opt/anaconda3/lib/python3.7/site-packages/dask/array/numpy_compat.py", line 41, in divide
# x = np.divide(x1, x2, out)
# File "cupy/core/core.pyx", line 1258, in cupy.core.core.ndarray.__array_ufunc__
# File "cupy/core/_kernel.pyx", line 811, in cupy.core._kernel.ufunc.__call__
# File "cupy/core/_kernel.pyx", line 89, in cupy.core._kernel._preprocess_args
# TypeError: Unsupported type <class 'numpy.ndarray'>


# Note: it works if axis=None
