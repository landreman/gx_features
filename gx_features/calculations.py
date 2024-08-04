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


def compute_mean_k_parallel(data, names, include_argmax=False):
    """Compute the mean k_parallel for each data point.

    data should have shape (n_data, n_z, n_quantities).

    If include_argmax=False, the function returns an array of shape (n_data,
    n_quantities).
    
    If include_argmax=True, the function returns an array of shape (n_data, n_quantities*2).
    """
    n_data, n_z, n_quantities = data.shape

    # Compute the Fourier transform of the data
    data_hat = np.fft.fft(data, axis=1)

    # Extract only the positive frequencies. Negative frequencies just have
    # amplitudes that are complex conjugates of the positive frequencies. Also
    # we are not interested in the DC component.
    max_index = n_z // 2
    data_hat = data_hat[:, 1:max_index, :]
    # k_parallel = k_parallel[1:max_index]
    k_parallel = np.arange(1, max_index)

    data_hat_abs = np.abs(data_hat)
    mean_k_parallel = np.sum(
        k_parallel[None, :, None] * data_hat_abs, axis=1
    ) / np.sum(data_hat_abs, axis=1)

    # To avoid NaNs:
    assert (
        np.min(np.sum(data_hat_abs, axis=1)) > 0
    ), "Some row of data was constant so mean k|| is not defined."

    mean_kpar_names = [n + "__mean_kpar" for n in names]

    if not include_argmax:
        return mean_k_parallel, mean_kpar_names
    
    argmax_k_parallel = np.argmax(data_hat_abs, axis=1) + 1
    argmax_kpar_names = [n + "__argmax_kpar" for n in names]

    new_features = np.concatenate((mean_k_parallel, argmax_k_parallel), axis=1)
    new_names = mean_kpar_names + argmax_kpar_names
    return new_features, new_names
