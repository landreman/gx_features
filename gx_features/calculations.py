import numpy as np
from scipy.stats import skew
import pandas as pd
from memory_profiler import profile


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

    # Handle the case in which data_hat_abs is all zero, so we just divided by 0.
    for j in range(n_quantities):
        sum_abs_data_hat = np.sum(data_hat_abs[:, :, j], axis=1) < 1e-13
        n_zeros = np.sum(sum_abs_data_hat)
        if n_zeros > 0:
            print("Warning!!! data_hat_abs = 0 for", n_zeros, "rows of", names[j])
        zero_locations = np.nonzero(np.sum(data_hat_abs[:, :, j], axis=1) < 1e-13)[0]
        if n_zeros > 0 and n_zeros < 100:
            print(zero_locations)
        mean_k_parallel[zero_locations, j] = 0

    # To avoid NaNs:
    # assert (
    #     np.min(np.sum(data_hat_abs, axis=1)) > 0
    # ), "Some row of data was constant so mean k|| is not defined."

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


def compute_mask_for_longest_true_interval(data):
    """Compute the longest interval of True values for each quantity.

    It only makes sense to apply this function after masks have been applied.

    data should have shape (n_data, n_z, n_quantities) and have dtype=bool.
    """
    print("Beginning compute_mask_for_longest_true_interval")
    n_data, n_z, n_quantities = data.shape

    zero_padding = np.zeros((n_data, 1, n_quantities))
    nonzeros3 = np.concatenate((zero_padding, data, data, data, zero_padding), axis=1)
    diff = nonzeros3[:, 1:, :] - nonzeros3[:, :-1, :]
    # np.set_printoptions(linewidth=400)

    features = np.zeros((n_data, n_z, n_quantities))
    for j_data in range(n_data):
        for j_quantity in range(n_quantities):
            starts = np.nonzero(diff[j_data, :, j_quantity] > 0.5)[0]
            ends = np.nonzero(diff[j_data, :, j_quantity] < -0.5)[0]
            interval_lengths = ends - starts
            # print(f"\nj_data: {j_data}, j_quantity: {j_quantity}")
            # print("data:")
            # print(data[j_data, :, j_quantity])
            # print("nonzeros3:")
            # print(nonzeros3[j_data, :, j_quantity])
            # print("diff:")
            # print(diff[j_data, :, j_quantity])
            # print("starts:", starts)
            # print("ends:  ", ends)
            assert len(starts) == len(ends)
            # If the array happens to be all 0's for a certain data entry,
            # starts and ends will be [], so max will fail.
            if len(starts) == 0:
                continue

            j_longest_interval = np.argmax(interval_lengths)
            start = starts[j_longest_interval] + 1
            end = ends[j_longest_interval] + 1
            # print("j_longest_interval:", j_longest_interval, "start:", start, "end:", end)

            arr = np.zeros(3 * n_z + 2)
            arr[start:end] = 1

            arr2 = arr[1:-1]
            arr3 = np.roll(arr2, n_z)
            arr4 = np.roll(arr3, n_z)
            arr5 = np.logical_or(np.logical_or(arr2, arr3), arr4)
            # print("arr: ", arr)
            # print("arr2:", arr2)
            # print("arr3:", arr3)
            # print("arr4:", arr4)
            # print("arr5:", arr5)
            features[j_data, :, j_quantity] = arr5[:n_z]

    print("Done with compute_mask_for_longest_true_interval")
    return features


def compute_max_minus_min(data, names):
    n_data, n_z, n_quantities = data.shape

    features = np.max(data, axis=1) - np.min(data, axis=1)
    new_names = [n + "__maxMinusMin" for n in names]

    return features, new_names

@profile
def compute_reductions(
    tensor,
    names,
    max=False,
    min=False,
    max_minus_min=False,
    mean=False,
    median=False,
    variance=False,
    rms=False,
    skewness=False,
    quantiles=None,
    count_above=None,
    fft_coefficients=None,
    mean_kpar=False,
    argmax_kpar=False,
):
    print("Beginning compute_reductions")
    n_data, n_z, n_quantities = tensor.shape
    assert len(names) == n_quantities

    n_features = 0
    if max:
        n_features += 1
    if min:
        n_features += 1
    if max_minus_min:
        n_features += 1
    if mean:
        n_features += 1
    if median:
        n_features += 1
    if variance:
        n_features += 1
    if rms:
        n_features += 1
    if skewness:
        n_features += 1
    if quantiles is not None:
        n_features += len(quantiles)
    if count_above is not None:
        n_features += len(count_above)
    if fft_coefficients is not None:
        n_features += len(fft_coefficients)
    if mean_kpar:
        n_features += 1
    if argmax_kpar:
        n_features += 1

    n_features_total = n_features * n_quantities
    features = np.zeros((n_data, n_features_total))
    index = 0
    new_names = []

    if max:
        features[:, index : index + n_quantities] = np.max(tensor, axis=1)
        new_names += [n + "_max" for n in names]
        index += n_quantities

    if min:
        features[:, index : index + n_quantities] = np.min(tensor, axis=1)
        new_names += [n + "_min" for n in names]
        index += n_quantities

    if max_minus_min:
        features[:, index : index + n_quantities] = np.max(tensor, axis=1) - np.min(
            tensor, axis=1
        )
        new_names += [n + "_maxMinusMin" for n in names]
        index += n_quantities

    if mean:
        features[:, index : index + n_quantities] = np.mean(tensor, axis=1)
        new_names += [n + "_mean" for n in names]
        index += n_quantities

    if median:
        features[:, index : index + n_quantities] = np.median(tensor, axis=1)
        new_names += [n + "_median" for n in names]
        index += n_quantities
        print("Done with median calculation")

    if rms:
        features[:, index : index + n_quantities] = np.sqrt(np.mean(tensor**2, axis=1))
        new_names += [n + "_rms" for n in names]
        index += n_quantities
        print("Done with RMS calculation")

    if variance:
        features[:, index : index + n_quantities] = np.var(tensor, axis=1)
        new_names += [n + "_variance" for n in names]
        index += n_quantities
        print("Done with variance calculation")

    if skewness:
        skew_data = skew(tensor, axis=1, bias=False)
        # If any of the quantities are constant, skewness will be NaN.
        features[:, index : index + n_quantities] = np.nan_to_num(skew_data)
        new_names += [n + "_skewness" for n in names]
        index += n_quantities
        print("Done with skewness calculation")

    if quantiles is not None:
        quantiles_result = np.quantile(tensor, quantiles, axis=1)
        for j_quantile, q in enumerate(quantiles):
            features[:, index : index + n_quantities] = quantiles_result[j_quantile, :]
            new_names += [n + f"_quantile{q}" for n in names]
            index += n_quantities
        print("Done with quantiles calculation")

    if count_above is not None:
        for t in count_above:
            features[:, index : index + n_quantities] = np.mean(tensor > t, axis=1)
            new_names += [n + f"_countAbove{t}" for n in names]
            index += n_quantities
        print("Done with count_above calculation")

    if fft_coefficients is not None:
        fft_result = np.fft.fft(tensor, axis=1)
        abs_fft_result = np.abs(fft_result)
        for j in fft_coefficients:
            features[:, index : index + n_quantities] = abs_fft_result[:, j, :]
            new_names += [n + f"_absFftCoeff{j}" for n in names]
            index += n_quantities
        print("Done with fft_coefficients calculation")

    if mean_kpar or argmax_kpar:
        new_features, kpar_names = compute_mean_k_parallel(
            tensor, names, include_argmax=True
        )
        assert len(kpar_names) == 2 * n_quantities
        if mean_kpar:
            features[:, index : index + n_quantities] = new_features[:, :n_quantities]
            new_names += kpar_names[:n_quantities]
            index += n_quantities
            print("Done with mean_kpar calculation")
        if argmax_kpar:
            features[:, index : index + n_quantities] = new_features[:, n_quantities:]
            new_names += kpar_names[n_quantities:]
            index += n_quantities
            print("Done with argmax_kpar calculation")

    assert len(new_names) == features.shape[1]
    assert index == n_features_total
    df = pd.DataFrame(features, columns=new_names)
    print("Done with compute_reductions")
    return df
