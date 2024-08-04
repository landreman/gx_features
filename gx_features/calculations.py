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
    mean_k_parallel = np.sum(k_parallel[None, :, None] * data_hat_abs, axis=1) / np.sum(
        data_hat_abs, axis=1
    )

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


def compute_longest_nonzero_interval(data, names):
    """Compute the longest interval of nonzero values for each quantity.

    It only makes sense to apply this function after masks have been applied.

    data should have shape (n_data, n_z, n_quantities).
    """
    n_data, n_z, n_quantities = data.shape

    # Determine how many quantities have both zero and nonzero values.
    quantities_to_consider = []
    new_names = []
    for j in range(n_quantities):
        print("data for quantity", j)
        print(data[:, :, j])
        if np.any(np.abs(data[:, :, j]) > 1e-13) and np.any(
            np.abs(data[:, :, j]) < 1e-13
        ):
            quantities_to_consider.append(j)
            new_names.append(names[j] + "__longestNon0Interval")

    quantities_to_consider = np.array(quantities_to_consider, dtype=int)
    n_quantities_to_consider = len(quantities_to_consider)

    # Pick out only the quantities that have both nonzero and zero values.
    nonzeros = np.abs(data[:, :, quantities_to_consider]) > 1e-13
    zero_padding = np.zeros((n_data, 1, n_quantities_to_consider))
    nonzeros3 = np.concatenate(
        (zero_padding, nonzeros, nonzeros, nonzeros, zero_padding), axis=1
    )
    diff = nonzeros3[:, 1:, :] - nonzeros3[:, :-1, :]

    features = np.zeros((n_data, n_quantities_to_consider))
    for j_data in range(n_data):
        for j_quantity in range(n_quantities_to_consider):
            starts = np.nonzero(diff[j_data, :, j_quantity] > 0.5)[0]
            ends = np.nonzero(diff[j_data, :, j_quantity] < -0.5)[0]
            interval_lengths = ends - starts
            # print(f"j_data: {j_data}, j_quantity: {j_quantity}")
            # print("nonzeros:")
            # print(nonzeros[j_data, :, j_quantity])
            # print("nonzeros3:")
            # print(nonzeros3[j_data, :, j_quantity])
            # print("diff:")
            # print(diff[j_data, :, j_quantity])
            # print("starts:", starts)
            # print("ends:  ", ends)
            assert len(starts) == len(ends)
            # If the array happens to be all 0's for a certain data entry,
            # starts and ends will be [], so max will fail.
            if len(starts) > 0:
                features[j_data, j_quantity] = np.max(interval_lengths)

    return features, new_names

def compute_max_minus_min(data, names):
    n_data, n_z, n_quantities = data.shape

    features = np.max(data, axis=1) - np.min(data, axis=1)
    new_names = [n + "__maxMinusMin" for n in names]

    return features, new_names