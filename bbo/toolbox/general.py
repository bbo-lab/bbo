import numpy as np

def nan_unwrap(x, axis=None):
    x = x.copy()
    x = np.mod(x+np.pi, 2*np.pi)-np.pi
    if axis is None:
        nonnanmask = ~np.isnan(x)
        x[nonnanmask] = np.unwrap(x[nonnanmask])
    else:
        nonnanmask = np.any(~np.isnan(x), axis=axis)
        x[nonnanmask] = np.unwrap(x[nonnanmask], axis=axis)
    return x
