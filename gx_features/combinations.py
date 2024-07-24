import numpy as np

def add_local_shear(feature_tensor, include_integral=True):
    """Adds the local shear and (optionally) the integrated local shear to the
    data."""
    # Shape of feature tensor: (n_data, n_z, n_quantities)
    assert feature_tensor.ndim == 3
    assert feature_tensor.shape[2] == 7

    n_data, n_z, n_quantities = feature_tensor.shape

    n_quantities_new = n_quantities + 1
    if include_integral:
        n_quantities_new += 1

    new_feature_tensor = np.zeros((n_data, n_z, n_quantities_new))
    new_feature_tensor[:, :, :n_quantities] = feature_tensor
    # z_functions: ['bmag', 'gbdrift', 'cvdrift', 'gbdrift0_over_shat', 'gds2', 'gds21_over_shat', 'gds22_over_shat_squared']
    integrated_local_shear = feature_tensor[:, :, 5] / feature_tensor[:, :, 6]

    new_feature_tensor[:, :, n_quantities] = np.gradient(feature_tensor[:, :, 0], axis=1)