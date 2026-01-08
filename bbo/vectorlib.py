import inspect
import logging
import numpy as np

logger = logging.getLogger(__name__)



def get_array_module(array):
    """
    Return the array module (np or cp) corresponding to the given array.

    - Returns `numpy` if it's a NumPy array or similar.
    - Returns `cupy` if it's a CuPy array, importing it lazily.
    - Uses the Array API `__array_namespace__` protocol if available.
    - Does NOT import CuPy unless the array is actually a CuPy array.
    """
    # 1. Use Array API standard if implemented
    if hasattr(array, "__array_namespace__"):
        return array.__array_namespace__()

    # 2. Fallback: check the module name
    mod = inspect.getmodule(type(array))
    if mod is not None and mod.__name__.startswith("cupy"):
        import cupy as cp
        return cp

    # 3. Default to numpy
    return np



def convert(img, module, dtype=None):
    if isinstance(img, str):
        return img
    if module == None:
        return img
    if isinstance(img, list):
        return [convert(i, module, dtype=dtype) for i in img]
    t = type(img)
    if inspect.getmodule(t) == module:
        return img
    if logging.DEBUG >= logging.root.level:
        finfo = inspect.getouterframes(inspect.currentframe())[1]
        logger.log(logging.DEBUG,
                   F'convert {t.__module__} to {module.__name__} by {finfo.filename} line {finfo.lineno}')
    if t.__module__ == 'cupy':
        return module.array(img.get(), copy=False, dtype=dtype)
    return module.array(img, copy=False, dtype=dtype)


def close_loop(data_array, axis=0):
    """
    Add the first point of `data_array` to the end along the specified axis
    to "close the loop".

    Parameters
    ----------
    data_array : np.ndarray or compatible (e.g., cupy.ndarray)
        The input array of data points.
    axes : int, optional
        The axis along which to close the loop (default is 0).

    Returns
    -------
    closed_array : same type as data_array
        The array with the first element appended at the end along `axes`.
    """
    # Handle both NumPy and CuPy by inferring xp from data_array
    xp = np if isinstance(data_array, np.ndarray) else __import__("cupy")

    first_point = xp.take(data_array, 0, axis=axis)
    closed_array = xp.concatenate([data_array, xp.expand_dims(first_point, axis=axis)], axis=axis)
    return closed_array
