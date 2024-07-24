import numpy as np

def differentiate(data):
    """Apply d/dz to the data, assuming periodic boundary conditions.
    
    Centered finite differences are used rather than a spectral method,
    due to the discontinuities in gds21 at the domain boundary.

    data should have shape (n_data, n_z).
    """
    n_z = data.shape[1]
    dz = 2 * np.pi / n_z
    return (np.roll(data, -1, axis=1) - np.roll(data, 1, axis=1)) / (2 * dz)
