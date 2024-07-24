import numpy as np

from .calculations import differentiate

def remove_cvdrift(feature_tensor):
    """Remove the cvdrift quantity from the feature tensor."""
    # Shape of feature tensor: (n_data, n_z, n_quantities)
    assert feature_tensor.ndim == 3
    assert feature_tensor.shape[2] == 7

    new_feature_tensor = np.zeros((feature_tensor.shape[0], feature_tensor.shape[1], 6))
    new_feature_tensor[:, :, :2] = feature_tensor[:, :, :2]
    new_feature_tensor[:, :, 2:] = feature_tensor[:, :, 3:]

    return new_feature_tensor


def add_local_shear(feature_tensor, include_integral=True):
    """Adds the local shear and (optionally) the integrated local shear to the
    data.
    
    If you are going to call remove_cvdrift(), it must be done before calling this function.
    """
    # Shape of feature tensor: (n_data, n_z, n_quantities)
    assert feature_tensor.ndim == 3
    assert feature_tensor.shape[2] == 7 or feature_tensor.shape[2] == 6

    n_data, n_z, n_quantities = feature_tensor.shape

    n_quantities_new = n_quantities + 1
    if include_integral:
        n_quantities_new += 1

    new_feature_tensor = np.zeros((n_data, n_z, n_quantities_new))
    new_feature_tensor[:, :, :n_quantities] = feature_tensor
    # z_functions: ['bmag', 'gbdrift', 'cvdrift', 'gbdrift0_over_shat', 'gds2', 'gds21_over_shat', 'gds22_over_shat_squared']
    integrated_local_shear = feature_tensor[:, :, -2] / feature_tensor[:, :, -1]

    new_feature_tensor[:, :, n_quantities] = differentiate(integrated_local_shear)
    if include_integral:
        new_feature_tensor[:, :, -1] = integrated_local_shear

    return new_feature_tensor