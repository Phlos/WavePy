from numba import jit
import numpy as np

from WavePyClasses import Grid, VectorThing

@jit
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

@jit
def compute_indices(loc_x, loc_z, X, Z):

    # based on an x and z location, calculate the indices at
    # which these are located
    # from https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array

    x = X[0,:]
    z = Z[:,0]
    idx_x, _ = find_nearest(x, loc_x)
    idx_z, _ = find_nearest(z, loc_z)

    return idx_z, idx_x