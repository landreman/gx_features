import os
import pickle
import psutil
import time
import gc
import numpy as np
from pandas import DataFrame, read_pickle
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
try:
    from tsfresh import extract_features
except:
    pass

import lightgbm as lgb
from memory_profiler import profile

from .io import load_all, load_tensor
from .calculations import (
    compute_mean_k_parallel,
    compute_max_minus_min,
    compute_reductions,
    compute_mask_for_longest_true_interval,
    differentiate,
)
from .combinations import (
    add_local_shear,
    create_masks,
    create_threshold_masks,
    make_pairwise_products_from_2_sets,
    make_feature_product_combinations,
    combine_tensors,
    make_feature_product_and_quotient_combinations,
    make_inverse_quantities,
    heaviside_transformations,
    divide_by_quantity,
)
from .mpi import distribute_work_mpi, join_dataframes_mpi
from .sequential_feature_selection import reductions_20241108
from .utils import (
    tensor_to_tsfresh_dataframe,
    drop_nearly_constant_features,
    drop_special_characters_from_column_names,
    simplify_names,
    meaningful_names,
)

n_logical_threads = psutil.cpu_count(logical=True)


def run_gc():
    # gc.collect(); gc.collect(); gc.collect()
    pass


def check_for_NaNs(extracted_features):
    """Check a DataFrame for any NaNs and report their locations."""
    nan_locations = np.nonzero(np.isnan(extracted_features.to_numpy()))
    if len(nan_locations[0]) > 0:
        columns = extracted_features.columns
        for j in range(len(nan_locations[0])):
            print(
                "NaN found at row",
                nan_locations[0][j],
                "and column",
                nan_locations[1][j],
                columns[nan_locations[1][j]],
            )
    else:
        print("No NaNs found")


def create_tensors_20240725_01(dataset):
    raw_tensor, raw_names, Y = load_tensor(dataset)

    # Create masks:
    masks, mask_names = create_masks(raw_tensor)

    # Add local shear as a feature:
    single_quantity_tensor, single_quantity_names = add_local_shear(
        raw_tensor, raw_names, include_integral=False
    )

    # Create feature-pair products:
    feature_product_tensor, feature_product_names = make_feature_product_combinations(
        single_quantity_tensor, single_quantity_names
    )

    # Create single-feature-mask combinations:
    feature_mask_tensor, feature_mask_names = make_pairwise_products_from_2_sets(
        single_quantity_tensor, single_quantity_names, masks, mask_names
    )

    # Create feature-pair-mask combinations:
    feature_pair_mask_tensor, feature_pair_mask_names = (
        make_pairwise_products_from_2_sets(
            feature_product_tensor, feature_product_names, masks, mask_names
        )
    )

    combinations_tensor, combinations_names = combine_tensors(
        feature_mask_tensor,
        feature_mask_names,
        feature_pair_mask_tensor,
        feature_pair_mask_names,
    )

    print("Number of combined quantities:", len(combinations_names))
    for n in combinations_names:
        print(n)

    return (
        single_quantity_tensor,
        single_quantity_names,
        combinations_tensor,
        combinations_names,
        Y,
    )


def create_features_20240726_01(dataset):
    (
        single_quantity_tensor,
        single_quantity_names,
        combinations_tensor,
        combinations_names,
        Y,
    ) = create_tensors_20240725_01(dataset)

    count_above_thresholds = np.arange(-2, 6.1, 0.5)
    count_above_params = [{"t": t} for t in count_above_thresholds]
    print("count_above_params:", count_above_params)

    fft_coefficients = []
    # Don't include the zeroth coefficient since this is just the mean,
    # which we will include separately.
    for j in range(1, 2):
        fft_coefficients.append({"attr": "abs", "coeff": j})
    print("fft_coefficients:", fft_coefficients)

    large_standard_deviation_params = [{"r": r} for r in np.arange(0.05, 1, 0.05)]
    print("large_standard_deviation_params:", large_standard_deviation_params)

    mean_n_absolute_max_params = [{"number_of_maxima": n} for n in range(2, 8)]
    print("mean_n_absolute_max_params:", mean_n_absolute_max_params)

    number_crossing_m_params = [{"m": m} for m in np.arange(-2, 2.5, 0.5)]
    print("number_crossing_m_params:", number_crossing_m_params)

    tsfresh_features_for_single_quantities = {
        "count_above": count_above_params,
        "fft_coefficient": fft_coefficients,
        "large_standard_deviation": large_standard_deviation_params,
        "maximum": None,
        "mean": None,
        "mean_n_absolute_max": mean_n_absolute_max_params,
        "median": None,
        "minimum": None,
        "quantile": [
            {"q": 0.1},
            {"q": 0.2},
            {"q": 0.3},
            {"q": 0.4},
            {"q": 0.6},
            {"q": 0.7},
            {"q": 0.8},
            {"q": 0.9},
        ],
        "ratio_beyond_r_sigma": [
            {"r": 0.5},
            {"r": 1},
            {"r": 1.5},
            {"r": 2},
            {"r": 2.5},
            {"r": 3},
            {"r": 5},
            {"r": 6},
            {"r": 7},
            {"r": 10},
        ],
        "variance": None,
    }
    # Other features to consider:
    # Features which are translation-invariant but which don't seem very intuitive
    # "binned_entropy": [{"max_bins": 10}],
    # "fft_aggregated": [{"aggtype": "centroid"}, {"aggtype": "variance"}, {'aggtype': 'skew'}, {'aggtype': 'kurtosis'}],
    # "skewness": None,
    # "kurtosis": None,
    # "fft_coefficient" with coefficients other than 1
    # "root_mean_square": None,
    # "count_above_mean": None,

    # abs_energy: This is identical to the autocorrelation at lag 0.
    # absolute_sum_of_changes: Not perfectly translation-invariant but nearly so
    # cid_ce: Not perfectly translation-invariant but nearly so
    # count_below: I think this doesn't provide more into than count_above
    # count_below_mean: I think this doesn't provide more into than count_above_mean
    # cwt_coefficients: wavelets sound useful, but not translation-invariant.
    # longest_strike_above_mean & longest_strike_below_mean: Not translation-invariant in tsfresh but I could
    # make a periodic version that is.
    # mean_abs_change: Not translation-invariant in tsfresh but I could make a
    # periodic version that is.
    # number_crossing_m: Not translation-invariant in tsfresh but I could make it so
    # number_peaks: Not translation-invariant in tsfresh but I could make it so.
    # standard_deviation: probably don't need both this and variance.
    # variance_larger_than_standard_deviation: unclear what this is for.

    single_quantity_df = tensor_to_tsfresh_dataframe(
        single_quantity_tensor, single_quantity_names
    )
    extracted_features_single_quantities = extract_features(
        single_quantity_df,
        column_id="j_tube",
        column_sort="z",
        default_fc_parameters=tsfresh_features_for_single_quantities,
    )
    print(
        "Number of tsfresh features from single quantities:",
        len(extracted_features_single_quantities.columns),
    )

    custom_features_array, custom_features_names = compute_mean_k_parallel(
        single_quantity_tensor, single_quantity_names
    )
    custom_features_df = DataFrame(custom_features_array, columns=custom_features_names)
    print("Number of custom features:", len(custom_features_names))

    # Now extract a smaller number of features from the masked combinations of quantities:

    tsfresh_features_for_combinations = {
        # "count_above": count_above_params,
        # "fft_coefficient": fft_coefficients,
        # "large_standard_deviation": large_standard_deviation_params,
        "maximum": None,
        "mean": None,
        # "mean_n_absolute_max": mean_n_absolute_max_params,
        # "median": None,
        "minimum": None,
    }
    combinations_df = tensor_to_tsfresh_dataframe(
        combinations_tensor, combinations_names
    )
    extracted_features_combinations = extract_features(
        combinations_df,
        column_id="j_tube",
        column_sort="z",
        default_fc_parameters=tsfresh_features_for_combinations,
    )
    print(
        "Number of tsfresh features from combinations:",
        len(extracted_features_combinations.columns),
    )

    # Find the features that are in the combinations but not in the single quantities:
    different_cols = extracted_features_combinations.columns.difference(
        extracted_features_single_quantities.columns
    )
    print(
        "Number of features in combinations but not in single quantities:",
        len(different_cols),
    )
    extracted_features = extracted_features_single_quantities.join(
        [custom_features_df, extracted_features_combinations[different_cols]]
    )

    print(
        "Number of features before dropping nearly constant features:",
        extracted_features.shape[1],
    )
    features = drop_nearly_constant_features(extracted_features)

    print("\n****** Final features ******\n")
    for f in features.columns:
        print(f)
    print("Final number of features:", features.shape[1])

    features["Y"] = Y
    drop_special_characters_from_column_names(features)

    if dataset == "test":
        filename = "20240601-01-kpar_and_pair_mask_features_test"
    elif dataset == "20240601":
        filename = "20240601-01-kpar_and_pair_mask_features"
    elif dataset == "20240726":
        filename = "20240726-01-kpar_and_pair_mask_features"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    features.to_pickle(filename + ".pkl")


def create_multithreshold_gbdrift_masked_gds22_features():
    raw_tensor, raw_names, Y = load_tensor("20240726")
    index = raw_names.index("gbdrift")
    assert index == 1
    gbdrift = raw_tensor[:, :, index]

    index = raw_names.index("gds22_over_shat_squared")
    assert index == 5
    raw_tensor = raw_tensor[:, :, index : index + 1]
    raw_names = raw_names[index : index + 1]

    # Create masks:
    masks, mask_names = create_threshold_masks(gbdrift)

    # Create single-feature-mask combinations:
    feature_mask_tensor, feature_mask_names = make_pairwise_products_from_2_sets(
        raw_tensor, raw_names, masks, mask_names
    )
    print("feature_mask_names:")
    for n in feature_mask_names:
        print(n)

    tsfresh_features_for_single_quantities = {
        "maximum": None,
        "mean": None,
        "median": None,
        "root_mean_square": None,
        "quantile": [
            {"q": 0.1},
            {"q": 0.2},
            {"q": 0.3},
            {"q": 0.4},
            {"q": 0.6},
            {"q": 0.7},
            {"q": 0.8},
            {"q": 0.9},
        ],
    }

    single_quantity_df = tensor_to_tsfresh_dataframe(
        feature_mask_tensor, feature_mask_names
    )
    extracted_features = extract_features(
        single_quantity_df,
        column_id="j_tube",
        column_sort="z",
        default_fc_parameters=tsfresh_features_for_single_quantities,
    )
    print(
        "Number of features before dropping nearly constant features:",
        extracted_features.shape[1],
    )
    features = drop_nearly_constant_features(extracted_features)

    print("\n****** Final features ******\n")
    for f in features.columns:
        print(f)
    print("Final number of features:", features.shape[1])

    features["Y"] = Y
    drop_special_characters_from_column_names(features)

    features.to_pickle("20240726_multithreshold_gbdrift_masked_gds22_features.pkl")


def create_features_20240730_01():
    (
        single_quantity_tensor,
        single_quantity_names,
        combinations_tensor,
        combinations_names,
        Y,
    ) = create_tensors_20240725_01("20240726")

    count_above_thresholds = np.arange(-2, 6.1, 0.5)
    count_above_params = [{"t": t} for t in count_above_thresholds]
    print("count_above_params:", count_above_params)

    fft_coefficients = []
    # Don't include the zeroth coefficient since this is just the mean,
    # which we will include separately.
    for j in range(1, 4):
        fft_coefficients.append({"attr": "abs", "coeff": j})
    print("fft_coefficients:", fft_coefficients)

    large_standard_deviation_params = [{"r": r} for r in np.arange(0.05, 1, 0.05)]
    print("large_standard_deviation_params:", large_standard_deviation_params)

    mean_n_absolute_max_params = [{"number_of_maxima": n} for n in range(2, 8)]
    print("mean_n_absolute_max_params:", mean_n_absolute_max_params)

    number_crossing_m_params = [{"m": m} for m in np.arange(-2, 2.5, 0.5)]
    print("number_crossing_m_params:", number_crossing_m_params)

    tsfresh_features_for_single_quantities = {
        "count_above": count_above_params,
        "fft_coefficient": fft_coefficients,
        "large_standard_deviation": large_standard_deviation_params,
        "maximum": None,
        "mean": None,
        "median": None,
        "minimum": None,
        "quantile": [
            {"q": 0.1},
            {"q": 0.2},
            {"q": 0.3},
            {"q": 0.4},
            {"q": 0.6},
            {"q": 0.7},
            {"q": 0.8},
            {"q": 0.9},
        ],
        # "ratio_beyond_r_sigma": [
        #     {"r": 0.5},
        #     {"r": 1},
        #     {"r": 1.5},
        #     {"r": 2},
        #     {"r": 2.5},
        #     {"r": 3},
        #     {"r": 5},
        #     {"r": 6},
        #     {"r": 7},
        #     {"r": 10},
        # ],
        "variance": None,
    }
    # Other features to consider:
    # Features which are translation-invariant but which don't seem very intuitive
    # "mean_n_absolute_max": mean_n_absolute_max_params,
    # "binned_entropy": [{"max_bins": 10}],
    # "fft_aggregated": [{"aggtype": "centroid"}, {"aggtype": "variance"}, {'aggtype': 'skew'}, {'aggtype': 'kurtosis'}],
    # "skewness": None,
    # "kurtosis": None,
    # "fft_coefficient" with coefficients other than 1
    # "root_mean_square": None,
    # "count_above_mean": None,

    # abs_energy: This is identical to the autocorrelation at lag 0.
    # absolute_sum_of_changes: Not perfectly translation-invariant but nearly so
    # cid_ce: Not perfectly translation-invariant but nearly so
    # count_below: I think this doesn't provide more into than count_above
    # count_below_mean: I think this doesn't provide more into than count_above_mean
    # cwt_coefficients: wavelets sound useful, but not translation-invariant.
    # longest_strike_above_mean & longest_strike_below_mean: Not translation-invariant in tsfresh but I could
    # make a periodic version that is.
    # mean_abs_change: Not translation-invariant in tsfresh but I could make a
    # periodic version that is.
    # number_crossing_m: Not translation-invariant in tsfresh but I could make it so
    # number_peaks: Not translation-invariant in tsfresh but I could make it so.
    # standard_deviation: probably don't need both this and variance.
    # variance_larger_than_standard_deviation: unclear what this is for.

    single_quantity_df = tensor_to_tsfresh_dataframe(
        single_quantity_tensor, single_quantity_names
    )
    extracted_features_single_quantities = extract_features(
        single_quantity_df,
        column_id="j_tube",
        column_sort="z",
        default_fc_parameters=tsfresh_features_for_single_quantities,
    )
    print(
        "Number of tsfresh features from single quantities:",
        len(extracted_features_single_quantities.columns),
    )

    custom_features_array, custom_features_names = compute_mean_k_parallel(
        single_quantity_tensor, single_quantity_names
    )
    custom_features_df = DataFrame(custom_features_array, columns=custom_features_names)
    print("Number of custom features:", len(custom_features_names))

    # Now extract a smaller number of features from the masked combinations of quantities:

    tsfresh_features_for_combinations = {
        # "count_above": count_above_params,
        # "fft_coefficient": fft_coefficients,
        # "large_standard_deviation": large_standard_deviation_params,
        "maximum": None,
        "mean": None,
        # "mean_n_absolute_max": mean_n_absolute_max_params,
        "median": None,
        "minimum": None,
        "root_mean_square": None,
        "quantile": [
            {"q": 0.1},
            {"q": 0.2},
            {"q": 0.3},
            {"q": 0.4},
            {"q": 0.6},
            {"q": 0.7},
            {"q": 0.8},
            {"q": 0.9},
        ],
        "variance": None,
    }
    combinations_df = tensor_to_tsfresh_dataframe(
        combinations_tensor, combinations_names
    )
    extracted_features_combinations = extract_features(
        combinations_df,
        column_id="j_tube",
        column_sort="z",
        default_fc_parameters=tsfresh_features_for_combinations,
    )
    print(
        "Number of tsfresh features from combinations:",
        len(extracted_features_combinations.columns),
    )

    # Find the features that are in the combinations but not in the single quantities:
    different_cols = extracted_features_combinations.columns.difference(
        extracted_features_single_quantities.columns
    )
    print(
        "Number of features in combinations but not in single quantities:",
        len(different_cols),
    )
    extracted_features = extracted_features_single_quantities.join(
        [custom_features_df, extracted_features_combinations[different_cols]]
    )

    print(
        "Number of features before dropping nearly constant features:",
        extracted_features.shape[1],
    )
    features = drop_nearly_constant_features(extracted_features)

    print("\n****** Final features ******\n")
    for f in features.columns:
        print(f)
    print("Final number of features:", features.shape[1])

    features["Y"] = Y
    drop_special_characters_from_column_names(features)

    filename = "20240726-01-kpar_and_pair_mask_features_20240730_01"

    features.to_pickle(filename + ".pkl")


def create_features_20240804_01(n_data=None):
    """
    If n_data is None, all data entries will be used.
    If n_data is an integer, the data will be trimmed to the first n_data entries.
    """
    raw_tensor, raw_names, Y = load_tensor("20240726")
    raw_names = simplify_names(raw_names)

    if n_data is not None:
        raw_tensor = raw_tensor[:n_data, :, :]
        Y = Y[:n_data]
    # raw_tensor = raw_tensor[3070:, :, :]
    # Y = Y[3070:]

    # Add local shear as a feature:
    F, F_names = add_local_shear(raw_tensor, raw_names, include_integral=False)

    # Add inverse quantities:
    inverse_tensor, inverse_names = make_inverse_quantities(F, F_names)
    F, F_names = combine_tensors(F, F_names, inverse_tensor, inverse_names)

    # CF, CF_names = make_feature_product_and_quotient_combinations(F, F_names)
    CF, CF_names = make_feature_product_combinations(F, F_names)

    M, M_names = heaviside_transformations(F, F_names)
    print("M_names:", M_names)

    MF, MF_names = make_pairwise_products_from_2_sets(F, F_names, M, M_names)
    MCF, MCF_names = make_pairwise_products_from_2_sets(CF, CF_names, M, M_names)

    tensor_before_inv_bmag, names_before_inv_bmag = combine_tensors(
        F, F_names, MF, MF_names, CF, CF_names, MCF, MCF_names
    )

    tensor_after_inv_bmag, names_after_inv_bmag = divide_by_quantity(
        tensor_before_inv_bmag, names_before_inv_bmag, raw_tensor[:, :, 0], "bmag"
    )

    tensor, names = combine_tensors(
        tensor_before_inv_bmag,
        names_before_inv_bmag,
        tensor_after_inv_bmag,
        names_after_inv_bmag,
    )

    print("\nQuantities before reduction:\n")
    for n in names:
        print(n)
    print("\nNumber of quantities before reduction:", len(names))

    ###########################################################################
    # Now apply reductions.
    ###########################################################################

    """
    # First compute any custom features:

    custom_features1, custom_features_names1 = compute_mean_k_parallel(
        tensor, names, include_argmax=True
    )
    custom_features2, custom_features_names2 = compute_max_minus_min(
        tensor, names,
    )
    custom_features = np.concatenate((custom_features1, custom_features2), axis=1)
    custom_features_names = custom_features_names1 + custom_features_names2
    custom_features_df = DataFrame(custom_features, columns=custom_features_names)
    print("Number of custom features:", len(custom_features_names))

    # Now compute the tsfresh features:

    count_above_thresholds = np.arange(-2, 6.1, 0.5)
    count_above_params = [{"t": t} for t in count_above_thresholds]
    print("count_above_params:", count_above_params)

    fft_coefficients = []
    # Don't include the zeroth coefficient since this is just the mean,
    # which we will include separately.
    for j in range(1, 4):
        fft_coefficients.append({"attr": "abs", "coeff": j})
    print("fft_coefficients:", fft_coefficients)

    tsfresh_feature_options = {
        "count_above": count_above_params,
        "fft_coefficient": fft_coefficients,
        "maximum": None,
        "mean": None,
        "median": None,
        "minimum": None,
        "quantile": [
            {"q": 0.1},
            {"q": 0.2},
            {"q": 0.3},
            {"q": 0.4},
            {"q": 0.6},
            {"q": 0.7},
            {"q": 0.8},
            {"q": 0.9},
        ],
        "root_mean_square": None,
        "skewness": None,
        "variance": None,
    }

    df_for_tsfresh = tensor_to_tsfresh_dataframe(tensor, names)
    print("n_jobs for tsfresh:", n_logical_threads)
    extracted_features_from_tsfresh = extract_features(
        df_for_tsfresh,
        column_id="j_tube",
        column_sort="z",
        default_fc_parameters=tsfresh_feature_options,
        n_jobs=n_logical_threads,
    )
    print(
        "Number of tsfresh features:",
        len(extracted_features_from_tsfresh.columns),
    )

    extracted_features = extracted_features_from_tsfresh.join(custom_features_df)
    """

    extracted_features = compute_reductions(
        tensor,
        names,
        max=True,
        min=True,
        max_minus_min=True,
        mean=True,
        median=True,
        rms=True,
        variance=True,
        skewness=True,
        quantiles=[0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
        count_above=np.arange(-2, 6.1, 0.5),
        fft_coefficients=[1, 2, 3],
        mean_kpar=True,
        argmax_kpar=True,
    )

    print(
        "Number of features before dropping nearly constant features:",
        extracted_features.shape[1],
    )
    check_for_NaNs(extracted_features)

    features = drop_nearly_constant_features(extracted_features)

    print("\n****** Final features ******\n")
    for f in features.columns:
        print(f)
    print("Final number of features:", features.shape[1])

    features["Y"] = Y
    drop_special_characters_from_column_names(features)

    filename = "20240726-01-kpar_and_pair_mask_features_20240804_01"

    features.to_pickle(filename + ".pkl")


def create_features_20240805_01(n_data=None):
    """
    These features are all variations on gds22 * Heaviside(gbdrift),
    attempting only to get an optimal 1st feature.

    If n_data is None, all data entries will be used.
    If n_data is an integer, the data will be trimmed to the first n_data entries.
    """
    raw_tensor, raw_names, Y = load_tensor("20240726")
    raw_names = simplify_names(raw_names)

    if n_data is not None:
        raw_tensor = raw_tensor[:n_data, :, :]
        Y = Y[:n_data]

    n_data, n_z, n_quantities = raw_tensor.shape

    index = 1
    gbdrift = raw_tensor[:, :, index]
    assert raw_names[index] == "gbdrift"

    index = 5
    gds22 = raw_tensor[:, :, index]
    assert raw_names[index] == "gds22"

    index = 0
    bmag = raw_tensor[:, :, index]
    assert raw_names[index] == "bmag"

    n_activation_functions = 6
    thresholds = np.arange(-1, 1.05, 0.1)
    n_thresholds = len(thresholds)
    activation_functions = np.zeros(
        (n_data, n_z, n_thresholds * n_activation_functions)
    )
    longest_true_interval_masks = compute_mask_for_longest_true_interval(
        gbdrift[:, :, None] - thresholds[None, None, :] > 0
    )

    activation_function_names = []
    for j_activation_function in range(n_activation_functions):
        for j_threshold, threshold in enumerate(thresholds):
            index = j_activation_function * n_thresholds + j_threshold
            x = gbdrift - threshold
            if j_activation_function == 0:
                activation_functions[:, :, index] = np.heaviside(x, 0)
                name = f"heaviside{threshold:.2f}"
            elif j_activation_function == 1:
                activation_functions[:, :, index] = 1 / (1 + np.exp(-x))
                name = f"sigmoid{threshold:.2f}"
            elif j_activation_function == 2:
                alpha = 0.05
                activation_functions[:, :, index] = alpha + (1 - alpha) * np.heaviside(
                    x, 0
                )
                name = f"leakyHeaviside{alpha:.2f}_{threshold:.2f}"
            elif j_activation_function == 3:
                alpha = 0.1
                activation_functions[:, :, index] = alpha + (1 - alpha) * np.heaviside(
                    x, 0
                )
                name = f"leakyHeaviside{alpha:.2f}_{threshold:.2f}"
            elif j_activation_function == 4:
                alpha = 0.2
                activation_functions[:, :, index] = alpha + (1 - alpha) * np.heaviside(
                    x, 0
                )
                name = f"leakyHeaviside{alpha:.2f}_{threshold:.2f}"
            elif j_activation_function == 5:
                activation_functions[:, :, index] = longest_true_interval_masks[
                    :, :, j_threshold
                ]
                name = f"longestGbdriftPosInterval{threshold:.2f}"
            else:
                raise RuntimeError("Should not get here")
            activation_function_names.append(name)
    print("activation_function_names:", activation_function_names)

    powers_of_gds22 = [0.5, 1, 2]
    powers_of_gbdrift = [
        0,
        1,
    ]  # Fractional powers disallowed because gbdrift can be <0.
    powers_of_bmag = [0, -1, -2]
    n_powers_of_gds22 = len(powers_of_gds22)
    n_powers_of_gbdrift = len(powers_of_gbdrift)
    n_powers_of_bmag = len(powers_of_bmag)

    powers_of_gds2_tensor = np.zeros((n_data, n_z, n_powers_of_gds22))
    powers_of_gbdrift_tensor = np.zeros((n_data, n_z, n_powers_of_gbdrift))
    powers_of_bmag_tensor = np.zeros((n_data, n_z, n_powers_of_bmag))
    powers_of_gds2_names = []
    powers_of_gbdrift_names = []
    powers_of_bmag_names = []
    for j_power, power in enumerate(powers_of_gds22):
        powers_of_gds2_tensor[:, :, j_power] = gds22**power
        powers_of_gds2_names.append(f"gds22^{power}")

    for j_power, power in enumerate(powers_of_gbdrift):
        powers_of_gbdrift_tensor[:, :, j_power] = gbdrift**power
        powers_of_gbdrift_names.append(f"gbdrift^{power}")

    for j_power, power in enumerate(powers_of_bmag):
        powers_of_bmag_tensor[:, :, j_power] = bmag**power
        powers_of_bmag_names.append(f"bmag^{power}")

    gbdrift_gds2_tensor, gbdrift_gds2_names = make_pairwise_products_from_2_sets(
        powers_of_gbdrift_tensor,
        powers_of_gbdrift_names,
        powers_of_gds2_tensor,
        powers_of_gds2_names,
    )
    print("gbdrift_gds2_names:", gbdrift_gds2_names)

    gbdrift_gds2_bmag_tensor, gbdrift_gds2_bmag_names = (
        make_pairwise_products_from_2_sets(
            gbdrift_gds2_tensor,
            gbdrift_gds2_names,
            powers_of_bmag_tensor,
            powers_of_bmag_names,
        )
    )

    tensor, names = make_pairwise_products_from_2_sets(
        gbdrift_gds2_bmag_tensor,
        gbdrift_gds2_bmag_names,
        activation_functions,
        activation_function_names,
    )

    print("\nQuantities before reduction:\n")
    for n in names:
        print(n)
    print("\nNumber of quantities before reduction:", len(names))

    ###########################################################################
    # Now apply reductions.
    ###########################################################################

    count_above_thresholds = np.arange(0.5, 10.1, 0.5)
    count_above_params = [{"t": t} for t in count_above_thresholds]
    print("count_above_params:", count_above_params)

    """
    tsfresh_feature_options = {
        "count_above": count_above_params,
        "maximum": None,
        "mean": None,
        "median": None,
        "quantile": [
            {"q": 0.1},
            {"q": 0.2},
            {"q": 0.3333},
            {"q": 0.6667},
            {"q": 0.8},
            {"q": 0.9},
        ],
        "root_mean_square": None,
        "variance": None,
    }

    df_for_tsfresh = tensor_to_tsfresh_dataframe(tensor, names)
    print("n_jobs for tsfresh:", n_logical_threads)
    extracted_features = extract_features(
        df_for_tsfresh,
        column_id="j_tube",
        column_sort="z",
        default_fc_parameters=tsfresh_feature_options,
        n_jobs=n_logical_threads,
    )
    print(
        "Number of tsfresh features:",
        len(extracted_features.columns),
    )
    """

    extracted_features = compute_reductions(
        tensor,
        names,
        max=True,
        mean=True,
        median=True,
        rms=True,
        variance=True,
        quantiles=[0.1, 0.2, 0.3333, 0.6667, 0.8, 0.9],
        count_above=count_above_thresholds,
    )

    print(
        "Number of features before dropping nearly constant features:",
        extracted_features.shape[1],
    )
    features = drop_nearly_constant_features(extracted_features)

    print("\n****** Final features ******\n")
    for f in features.columns:
        print(f)
    print("Final number of features:", features.shape[1])

    features["Y"] = Y
    drop_special_characters_from_column_names(features)

    filename = "20240726-01-gbdrift_gds22_combo_features_20240805_01"

    features.to_pickle(filename + ".pkl")


def create_features_20240906_01(n_data=None):
    """
    Same as 20240804_01, but for finite-beta rather than vacuum, so cvdrift is
    also included.

    If n_data is None, all data entries will be used.
    If n_data is an integer, the data will be trimmed to the first n_data entries.
    """
    raw_tensor, raw_names, Y = load_tensor("20240601")
    raw_names = simplify_names(raw_names)

    if n_data is not None:
        raw_tensor = raw_tensor[:n_data, :, :]
        Y = Y[:n_data]
    # raw_tensor = raw_tensor[3070:, :, :]
    # Y = Y[3070:]

    # Add local shear as a feature:
    F, F_names = add_local_shear(raw_tensor, raw_names, include_integral=False)

    # Add inverse quantities:
    inverse_tensor, inverse_names = make_inverse_quantities(F, F_names)
    F, F_names = combine_tensors(F, F_names, inverse_tensor, inverse_names)

    # CF, CF_names = make_feature_product_and_quotient_combinations(F, F_names)
    CF, CF_names = make_feature_product_combinations(F, F_names)

    M, M_names = heaviside_transformations(F, F_names)
    print("M_names:", M_names)

    MF, MF_names = make_pairwise_products_from_2_sets(F, F_names, M, M_names)
    MCF, MCF_names = make_pairwise_products_from_2_sets(CF, CF_names, M, M_names)

    tensor_before_inv_bmag, names_before_inv_bmag = combine_tensors(
        F, F_names, MF, MF_names, CF, CF_names, MCF, MCF_names
    )

    tensor_after_inv_bmag, names_after_inv_bmag = divide_by_quantity(
        tensor_before_inv_bmag, names_before_inv_bmag, raw_tensor[:, :, 0], "bmag"
    )

    tensor, names = combine_tensors(
        tensor_before_inv_bmag,
        names_before_inv_bmag,
        tensor_after_inv_bmag,
        names_after_inv_bmag,
    )

    print("\nQuantities before reduction:\n")
    for n in names:
        print(n)
    print("\nNumber of quantities before reduction:", len(names))

    ###########################################################################
    # Now apply reductions.
    ###########################################################################

    extracted_features = compute_reductions(
        tensor,
        names,
        max=True,
        min=True,
        max_minus_min=True,
        mean=True,
        median=True,
        rms=True,
        variance=True,
        skewness=True,
        quantiles=[0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
        count_above=np.arange(-2, 6.1, 0.5),
        fft_coefficients=[1, 2, 3],
        mean_kpar=True,
        argmax_kpar=True,
    )

    print(
        "Number of features before dropping nearly constant features:",
        extracted_features.shape[1],
    )
    check_for_NaNs(extracted_features)

    features = drop_nearly_constant_features(extracted_features)

    print("\n****** Final features ******\n")
    for f in features.columns:
        print(f)
    print("Final number of features:", features.shape[1])

    features["Y"] = Y
    drop_special_characters_from_column_names(features)

    filename = "20240601-01_features_20240906_01"

    features.to_pickle(filename + ".pkl")


# @profile
def create_features_20241011_01(n_data=None, mpi=False, dataset="20241005"):
    """
    Same as 20240804_01, but for finite-beta rather than vacuum, so cvdrift is
    also included. Also, the scalar features [nfp, iota, shat, d_pressure_d_s]
    are added.

    If n_data is None, all data entries will be used.
    If n_data is an integer, the data will be trimmed to the first n_data entries.
    """
    start_time = time.time()
    data = load_all(dataset)
    print(time.time() - start_time, "Done loading data", flush=True)
    raw_tensor = data["feature_tensor"]
    raw_names = data["z_functions"]
    Y = data["Y"]
    extra_scalar_features = data["scalar_feature_matrix"]

    raw_names = simplify_names(raw_names)
    print(time.time() - start_time, "Done simplifying names", flush=True)

    if n_data is not None:
        raw_tensor = raw_tensor[:n_data, :, :]
        Y = Y[:n_data]
        extra_scalar_features = extra_scalar_features[:n_data, :]

    if mpi:
        from .mpi import proc0_print

        raw_tensor, extra_scalar_features = distribute_work_mpi(
            raw_tensor, extra_scalar_features
        )
        from mpi4py import MPI

        proc0 = MPI.COMM_WORLD.rank == 0
    else:
        proc0_print = print
        proc0 = True

    # Add local shear as a feature:
    F, F_names = add_local_shear(raw_tensor, raw_names, include_integral=False)
    print(time.time() - start_time, "Done adding local shear", flush=True)

    # Add inverse quantities:
    inverse_tensor, inverse_names = make_inverse_quantities(F, F_names)
    print(time.time() - start_time, "Done creating inverse quantities", flush=True)
    F, F_names = combine_tensors(F, F_names, inverse_tensor, inverse_names)
    del inverse_tensor  # Free up memory
    print(
        time.time() - start_time,
        "Done combining inverse quantities with original raw features",
        flush=True,
    )

    CF, CF_names = make_feature_product_combinations(F, F_names)
    print(
        time.time() - start_time,
        "Done forming feature product combinations",
        flush=True,
    )

    M, M_names = heaviside_transformations(F, F_names)
    proc0_print(
        time.time() - start_time, "Done creating Heaviside transformations", flush=True
    )
    proc0_print("M_names:", M_names, flush=True)

    MF, MF_names = make_pairwise_products_from_2_sets(F, F_names, M, M_names)
    print(time.time() - start_time, "Done creating MF features", flush=True)
    MCF, MCF_names = make_pairwise_products_from_2_sets(CF, CF_names, M, M_names)
    print(time.time() - start_time, "Done creating MCF features", flush=True)

    tensor_before_inv_bmag, names_before_inv_bmag = combine_tensors(
        F, F_names, MF, MF_names, CF, CF_names, MCF, MCF_names
    )
    del F, MF, CF, MCF  # Free up memory
    print(
        time.time() - start_time,
        "Done combining tensors before dividing by bmag",
        flush=True,
    )

    tensor_after_inv_bmag, names_after_inv_bmag = divide_by_quantity(
        tensor_before_inv_bmag, names_before_inv_bmag, raw_tensor[:, :, 0], "bmag"
    )
    print(time.time() - start_time, "Done dividing by bmag", flush=True)

    tensor, names = combine_tensors(
        tensor_before_inv_bmag,
        names_before_inv_bmag,
        tensor_after_inv_bmag,
        names_after_inv_bmag,
    )
    del tensor_before_inv_bmag, tensor_after_inv_bmag  # Free up memory
    print(
        time.time() - start_time,
        "Done combining tensors with and without 1/B",
        flush=True,
    )

    proc0_print(time.time() - start_time, "\nQuantities before reduction:\n")
    for n in names:
        proc0_print(n)
    proc0_print("\nNumber of quantities before reduction:", len(names), flush=True)

    ###########################################################################
    # Now apply reductions.
    ###########################################################################

    run_gc()
    extracted_features = compute_reductions(
        tensor,
        names,
        max=True,
        min=True,
        max_minus_min=True,
        mean=True,
        median=True,
        rms=True,
        variance=True,
        skewness=True,
        quantiles=[0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
        count_above=np.arange(-2, 6.1, 0.5),
        fft_coefficients=[1, 2, 3],
        mean_kpar=True,
        argmax_kpar=True,
    )
    del tensor  # Free up memory

    proc0_print(
        time.time() - start_time,
        "Number of features before dropping nearly constant features:",
        extracted_features.shape[1],
        flush=True,
    )
    check_for_NaNs(extracted_features)

    ###########################################################################
    # Add extra scalar features
    ###########################################################################

    extra_scalar_features_df = DataFrame(extra_scalar_features, columns=data["scalars"])

    extracted_features = extracted_features.join(extra_scalar_features_df)

    ###########################################################################
    # Finishing touches
    ###########################################################################

    if mpi:
        extracted_features = join_dataframes_mpi(extracted_features)

    if proc0:
        features = drop_nearly_constant_features(extracted_features)

        proc0_print("\n****** Final features ******\n")
        for f in features.columns:
            proc0_print(f)
        proc0_print("Final number of features:", features.shape[1])

        features["Y"] = Y
        drop_special_characters_from_column_names(features)

        filename = dataset + "-01_features_20241011_01"
        features.to_pickle(filename + ".pkl")

        return features


def unary_funcs_20241119(index, arr_in, name_in, powers=[-1, 2], return_n_unary=False):
    # In the future we could also add np.roll by various amounts, or shifts
    # before Heaviside/ReLU.

    # The indefinite integral is another option, but the choice of
    # which z to start at breaks translation invariance.
    n_unitary_operations_besides_powers = 7
    if return_n_unary:
        return n_unitary_operations_besides_powers + len(powers)

    power_strings = {
        -1: "⁻¹",
        2: "²",
    }
    if index == 0:
        # Identity
        arr_out = arr_in
        name_out = name_in
    elif index == 1:
        # Absolute value
        arr_out = np.abs(arr_in)
        name_out = f"|{name_in}|"
    elif index == 2:
        # Derivative
        arr_out = differentiate(arr_in)
        name_out = f"∂({name_in})"
    elif index == 3:
        arr_out = np.heaviside(arr_in, 0)
        name_out = f"Heaviside({name_in})"
    elif index == 4:
        arr_out = np.heaviside(-arr_in, 0)
        name_out = f"Heaviside(-{name_in})"
    elif index == 5:
        arr_out = np.where(arr_in > 0, arr_in, 0)
        name_out = f"ReLU({name_in})"
    elif index == 6:
        arr_out = np.where(-arr_in > 0, -arr_in, 0)
        name_out = f"ReLU(-{name_in})"
    elif (
        index >= n_unitary_operations_besides_powers
        and index < n_unitary_operations_besides_powers + len(powers)
    ):
        power = powers[index - n_unitary_operations_besides_powers]
        if power < 0 and np.any(arr_in == 0):
            # Avoid division by zero
            arr_out = np.full_like(arr_in, np.nan)
        else:
            arr_out = arr_in**power

        name_out = f"({name_in}){power_strings[power]}"
    else:
        raise RuntimeError("Should not get here")

    # If the operation doesn't do anything, and we aren't explicitly asking for
    # the identity function, return NaN so the superfluous function is not considered.
    if index > 0 and np.array_equal(arr_in, arr_out):
        arr_out = np.full_like(arr_in, np.nan)

    return arr_out, name_out


def unary_funcs_20241123(index, arr_in, name_in, powers=[-1, 2], return_n_unary=False):
    # This is a small set of unary functions, for testing.

    n_unitary_operations_besides_powers = 2

    if return_n_unary:
        return n_unitary_operations_besides_powers + len(powers)

    power_strings = {
        -1: "⁻¹",
        2: "²",
    }
    if index == 0:
        # Identity
        arr_out = arr_in
        name_out = name_in
    elif index == 1:
        arr_out = np.heaviside(arr_in, 0)
        name_out = f"Heaviside({name_in})"
    elif (
        index >= n_unitary_operations_besides_powers
        and index < n_unitary_operations_besides_powers + len(powers)
    ):
        power = powers[index - n_unitary_operations_besides_powers]
        if power < 0 and np.any(arr_in == 0):
            # Avoid division by zero
            arr_out = np.full_like(arr_in, np.nan)
        else:
            arr_out = arr_in**power

        name_out = f"({name_in}){power_strings[power]}"
    else:
        raise RuntimeError("Should not get here")

    # If the operation doesn't do anything, and we aren't explicitly asking for
    # the identity function, return NaN so the superfluous function is not considered.
    if index > 0 and np.array_equal(arr_in, arr_out):
        arr_out = np.full_like(arr_in, np.nan)

    return arr_out, name_out


def compute_fn_20241119(
    data,
    mpi_rank,
    mpi_size,
    evaluator,
    unary_func=unary_funcs_20241119,
    reductions_func=reductions_20241108,
    algorithm=2,
    n_B_powers=2,
):
    z_functions = data["z_functions"]
    feature_tensor = data["feature_tensor"]
    scalars = data["scalars"]
    scalar_feature_matrix = data["scalar_feature_matrix"]

    n_scalars = len(scalars)
    n_data, n_z, n_quantities = feature_tensor.shape

    n_reductions = reductions_func(1, 1, return_n_reductions=True)

    n_unary = unary_func(None, None, None, return_n_unary=True)

    index = 0
    bmag = feature_tensor[:, :, index]
    assert z_functions[index] == "bmag"

    # Add local shear as a feature:
    F, F_names = add_local_shear(feature_tensor, z_functions, include_integral=False)
    n_F = len(F_names)
    F_names = meaningful_names(F_names)

    # Explicitly store U(F) for convenience.
    n_U_F_original = n_unary * n_F
    U_F = np.zeros((n_data, n_z, n_U_F_original))
    U_F_names = []
    index = 0
    for j_unary in range(n_unary):
        for j_F in range(n_F):
            arr, name = unary_func(j_unary, F[:, :, j_F], F_names[j_F])
            # Only store if there are no NaNs and the quantity is not constant.
            if np.max(arr) > np.min(arr) and (not np.isnan(arr).any()):
                U_F_names.append(name)
                U_F[:, :, index] = arr
                index += 1
            else:
                if mpi_rank == 0:
                    print(f"Skipping {name} because it is constant or has NaNs")

    n_U_F = index
    U_F = U_F[:, :, :n_U_F]

    # Tuples are (exponent, string)
    if n_B_powers == 1:
        extra_powers_of_B = [(0, "")]
    elif n_B_powers == 2:
        extra_powers_of_B = [(0, ""), (-1, " / B")]
    elif n_B_powers == 3:
        extra_powers_of_B = [(0, ""), (-1, " / B"), (-2, " / B²")]
    else:
        raise ValueError("Invalid n_B_powers")
    
    n_C = (n_U_F * (n_U_F + 1)) // 2 + n_U_F
    n_total = n_C * n_unary * n_reductions * len(extra_powers_of_B) + n_scalars
        
    if mpi_rank == 0:
        print("Any Nans?", np.any(np.isnan(U_F)))
        print("Any infs?", np.any(np.isinf(U_F)))
        print("names after adding local shear:", F_names)
        print("n_F:", n_F)
        print("n_unary:", n_unary)
        print("n_U_F_original:", n_U_F_original)
        print("n_U_F:", n_U_F)
        print("n_reductions:", n_reductions)
        print("n_scalars:", n_scalars)
        print("n_B_powers:", n_B_powers)
        print(
            "Total number of features that will be evaluated:",
            n_total,
            flush=True,
        )
        print(flush=True)
    # return

    from mpi4py import MPI
    MPI.COMM_WORLD.barrier()

    j_combos_1_arr = np.zeros(n_C, dtype=np.int32)
    j_combos_2_arr = np.zeros(n_C, dtype=np.int32)
    index = 0
    for j_combos_1 in range(n_U_F):
        for j_combos_2 in range(-1, j_combos_1 + 1):
            j_combos_1_arr[index] = j_combos_1
            j_combos_2_arr[index] = j_combos_2
            index += 1
    assert index == n_C

    index = 0
    # Try all the extra scalar features first:
    for j in range(n_scalars):
        evaluator(scalar_feature_matrix[:, j], scalars[j], index)
        index += 1

    # Now the main features:

    if algorithm == 1:
        if mpi_rank == 0:
            print("Using algorithm 1")
        for j_combos_1 in range(n_U_F):
            for j_combos_2 in range(-1, j_combos_1 + 1):
                # j_combos_2 = -1 means just use j_combos_1 without a product
                if j_combos_2 == -1:
                    C_U_F = U_F[:, :, j_combos_1]
                    C_U_F_name = U_F_names[j_combos_1]
                else:
                    C_U_F = U_F[:, :, j_combos_1] * U_F[:, :, j_combos_2]
                    C_U_F_name = f"{U_F_names[j_combos_1]} {U_F_names[j_combos_2]}"

                for j_outer_unary in range(n_unary):
                    U_C_U_F, U_C_U_F_name = unary_func(j_outer_unary, C_U_F, C_U_F_name)
                    if (not np.all(np.isfinite(U_C_U_F))) or np.max(arr) == np.min(arr):
                        continue

                    for extra_power_of_B, extra_power_of_B_str in extra_powers_of_B:
                        B_U_C_U_F = U_C_U_F * bmag**extra_power_of_B
                        B_U_C_U_F_name = f"{U_C_U_F_name}{extra_power_of_B_str}"
                        for j_reduction in range(n_reductions):
                            if index % mpi_size == mpi_rank:
                                # Only evaluate if this proc needs to:

                                reduction, reduction_name = reductions_func(
                                    B_U_C_U_F, j_reduction
                                )

                                final_name = f"{reduction_name}({B_U_C_U_F_name})"
                                if index % 1000 == 0:
                                    print("Progress:", index, final_name, flush=True)
                                evaluator(reduction, final_name, index)

                            index += 1
    elif algorithm == 2:
        # algorithm 2
        if mpi_rank == 0:
            print("Using algorithm 2")
            print(
                "n_C:",
                n_C,
                "and each proc will do",
                n_C // mpi_size,
                "of these",
            )
        index_for_evaluator = (
            mpi_rank  # so the evaluator will always evaluate the cost function
        )

        index = 0
        for j_outer in range(mpi_rank, n_C, mpi_size):
            j_combos_1 = j_combos_1_arr[j_outer]
            j_combos_2 = j_combos_2_arr[j_outer]
            # j_combos_2 = -1 means just use j_combos_1 without a product
            if j_combos_2 == -1:
                C_U_F = U_F[:, :, j_combos_1]
                C_U_F_name = U_F_names[j_combos_1]
            else:
                C_U_F = U_F[:, :, j_combos_1] * U_F[:, :, j_combos_2]
                C_U_F_name = f"{U_F_names[j_combos_1]} {U_F_names[j_combos_2]}"
            if (j_outer - mpi_rank) % (100 * mpi_size) == 0:
                print(
                    "rank",
                    mpi_rank,
                    "is starting j_outer",
                    j_outer,
                    "of",
                    n_C,
                    C_U_F_name,
                    flush=True,
                )

            for j_outer_unary in range(n_unary):
                U_C_U_F, U_C_U_F_name = unary_func(j_outer_unary, C_U_F, C_U_F_name)
                if (not np.all(np.isfinite(U_C_U_F))) or np.max(arr) == np.min(arr):
                    continue

                for extra_power_of_B, extra_power_of_B_str in extra_powers_of_B:
                    B_U_C_U_F = U_C_U_F * bmag**extra_power_of_B
                    B_U_C_U_F_name = f"{U_C_U_F_name}{extra_power_of_B_str}"
                    for j_reduction in range(n_reductions):
                        reduction, reduction_name = reductions_func(
                            B_U_C_U_F, j_reduction
                        )

                        final_name = f"{reduction_name}({B_U_C_U_F_name})"
                        if index % 1000 == 0:
                            print(
                                f"Progress: rank {mpi_rank:4} has done {index} evals",
                                final_name,
                                flush=True,
                            )
                        evaluator(reduction, final_name, index_for_evaluator)
                        index += 1
    elif algorithm == 3:
        # algorithm 3
        n_outer = n_C * n_unary
        if mpi_rank == 0:
            print("Using algorithm 3")
            print(
                "n_outer:",
                n_outer,
                "and each proc will do",
                n_outer // mpi_size,
                "of these",
            )
        index_for_evaluator = (
            mpi_rank  # so the evaluator will always evaluate the cost function
        )

        index = 0
        for j_outer in range(mpi_rank, n_outer, mpi_size):
            j_C = j_outer // n_unary
            j_outer_unary = j_outer % n_unary

            j_combos_1 = j_combos_1_arr[j_C]
            j_combos_2 = j_combos_2_arr[j_C]
            # j_combos_2 = -1 means just use j_combos_1 without a product
            if j_combos_2 == -1:
                C_U_F = U_F[:, :, j_combos_1]
                C_U_F_name = U_F_names[j_combos_1]
            else:
                C_U_F = U_F[:, :, j_combos_1] * U_F[:, :, j_combos_2]
                C_U_F_name = f"{U_F_names[j_combos_1]} {U_F_names[j_combos_2]}"

            if (j_outer - mpi_rank) % (100 * mpi_size) == 0:
                print(
                    "rank",
                    mpi_rank,
                    "is starting j_outer",
                    j_outer,
                    "of",
                    n_outer,
                    C_U_F_name,
                    flush=True,
                )

            U_C_U_F, U_C_U_F_name = unary_func(j_outer_unary, C_U_F, C_U_F_name)
            if (not np.all(np.isfinite(U_C_U_F))) or np.max(arr) == np.min(arr):
                continue

            for extra_power_of_B, extra_power_of_B_str in extra_powers_of_B:
                B_U_C_U_F = U_C_U_F * bmag**extra_power_of_B
                B_U_C_U_F_name = f"{U_C_U_F_name}{extra_power_of_B_str}"
                for j_reduction in range(n_reductions):
                    reduction, reduction_name = reductions_func(
                        B_U_C_U_F, j_reduction
                    )

                    final_name = f"{reduction_name}({B_U_C_U_F_name})"
                    if index % 1000 == 0:
                        print(
                            f"Progress: rank {mpi_rank:4} has done {index} evals",
                            final_name,
                            flush=True,
                        )
                    evaluator(reduction, final_name, index_for_evaluator)
                    index += 1
    else:
        raise ValueError("Invalid algorithm")

def reductions_20241129(
    arr,
    j,
    return_n_reductions=False,
):
    """These are reductions that seem likely to turn up for the top single feature."""
    n_reductions = 4
    if return_n_reductions:
        return n_reductions

    if j == 0:
        return arr.max(axis=1), "max"

    elif j == 1:
        return arr.mean(axis=1), "mean"

    elif j == 2:
        return np.sqrt(np.mean(arr**2, axis=1)), "rootMeanSquare"

    elif j == 3:
        return np.var(arr, axis=1), "variance"

    else:
        raise ValueError(f"Invalid reduction index: {j}")
    
def compute_fn_20241129(data, mpi_rank, mpi_size, evaluator, reductions_func=reductions_20241129):
    """
    Focus on just the first feature, and try quantities of the form

    reduction[(Heaviside(cvdrift + thresh) + alpha * cvdrift) * gds22^power * B^power]

    See create_features_20240805_01() for a similar set of features.
    """
    z_functions = data["z_functions"]
    feature_tensor = data["feature_tensor"]

    n_reductions = reductions_func(1, 1, return_n_reductions=True)

    n_data, n_z, n_quantities = feature_tensor.shape

    index = 2
    cvdrift = feature_tensor[:, :, index]
    assert z_functions[index] == "cvdrift"

    index = 6
    gds22 = feature_tensor[:, :, index]
    assert z_functions[index] == "gds22_over_shat_squared"

    index = 0
    bmag = feature_tensor[:, :, index]
    assert z_functions[index] == "bmag"

    z_functions = meaningful_names(z_functions)

    alphas = [0, 0.05, 0.1, 0.2]
    n_alphas = len(alphas)

    thresholds = np.arange(-0.5, 0.55, 0.1)
    n_thresholds = len(thresholds)

    powers_of_gds22 = [0.5, 1, 2]
    n_powers_of_gds22 = len(powers_of_gds22)

    powers_of_bmag = [0, -1, -1.5, -2, -2.5, -3, -3.5]
    n_powers_of_bmag = len(powers_of_bmag)

    n_total = n_alphas * n_thresholds * n_powers_of_gds22 * n_powers_of_bmag * n_reductions

    if mpi_rank == 0:
        print("n_alphas:", n_alphas)
        print("n_thresholds:", n_thresholds)
        print("n_powers_of_bmag:", n_powers_of_bmag)
        print("n_powers_of_gds22:", n_powers_of_gds22)
        print("n_reductions:", n_reductions)
        print("Total number of features to consider:", n_total)
        print(flush=True)
    # return

    from mpi4py import MPI
    MPI.COMM_WORLD.barrier()

    for j_total in range(mpi_rank, n_total, mpi_size):
        j_reduction = j_total % n_reductions
        j_rest = j_total // n_reductions

        j_power_of_bmag = j_rest % n_powers_of_bmag
        j_rest = j_rest // n_powers_of_bmag

        j_power_of_gds22 = j_rest % n_powers_of_gds22
        j_rest = j_rest // n_powers_of_gds22

        j_threshold = j_rest % n_thresholds
        j_rest = j_rest // n_thresholds

        j_alpha = j_rest

        threshold = thresholds[j_threshold]
        alpha = alphas[j_alpha]
        power_of_gds22 = powers_of_gds22[j_power_of_gds22]
        power_of_bmag = powers_of_bmag[j_power_of_bmag]

        data = (
            (np.heaviside(cvdrift - threshold, 0) + alpha * cvdrift)
            * gds22**power_of_gds22
            * bmag**power_of_bmag
        )
        name = (
            f"[Heaviside(B⁻²𝗕×κ⋅∇y - {threshold:g}) + {alpha} B⁻²𝗕×κ⋅∇y]"
            f" |∇x|²^{power_of_gds22}"
            f" B^{power_of_bmag}"
        )
        reduction, reduction_name = reductions_func(
            data, j_reduction
        )
        # In evaluator, set index = mpi_rank so the evaluator will always evaluate the cost function
        evaluator(reduction, f"{reduction_name}({name})", mpi_rank)

        if j_total % 1000 == 0:
            print(f"index: {j_total}  name: {name}", flush=True)


def compute_fn_20241130(data, mpi_rank, mpi_size, evaluator, reductions_func=reductions_20241129):
    """
    Focus on just the first feature, and try quantities of the form

    reduction[(Heaviside(cvdrift + thresh) + gamma * ReLU(cvdrift + thresh) + alpha * cvdrift + beta) * gds22^power * B^power]

    See create_features_20240805_01() for a similar set of features.
    """
    z_functions = data["z_functions"]
    feature_tensor = data["feature_tensor"]

    n_reductions = reductions_func(1, 1, return_n_reductions=True)

    n_data, n_z, n_quantities = feature_tensor.shape

    index = 2
    cvdrift = feature_tensor[:, :, index]
    assert z_functions[index] == "cvdrift"

    index = 6
    gds22 = feature_tensor[:, :, index]
    assert z_functions[index] == "gds22_over_shat_squared"

    index = 0
    bmag = feature_tensor[:, :, index]
    assert z_functions[index] == "bmag"

    z_functions = meaningful_names(z_functions)

    alphas = [0, 0.05, 0.1, 0.2, 0.3, 0.4]
    n_alphas = len(alphas)

    gammas = [0, 0.05, 0.1, 0.2, 0.3, 0.4]
    n_gammas = len(gammas)

    betas = [0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3, 1.7]
    n_betas = len(betas)

    #thresholds = np.arange(-0.5, 0.55, 0.1)
    thresholds = [-0.1, 0, 0.1]
    n_thresholds = len(thresholds)

    powers_of_gds22 = [0.5, 1, 1.5, 2, 2.5]
    n_powers_of_gds22 = len(powers_of_gds22)

    powers_of_bmag = [0, -1, -2, -2.5, -3, -3.5, -4, -4.5]
    n_powers_of_bmag = len(powers_of_bmag)

    n_total = n_alphas * n_betas * n_gammas * n_thresholds * n_powers_of_gds22 * n_powers_of_bmag * n_reductions

    if mpi_rank == 0:
        print("n_alphas:", n_alphas)
        print("n_betas:", n_betas)
        print("n_gammas:", n_gammas)
        print("n_thresholds:", n_thresholds)
        print("n_powers_of_bmag:", n_powers_of_bmag)
        print("n_powers_of_gds22:", n_powers_of_gds22)
        print("n_reductions:", n_reductions)
        print("Total number of features to consider:", n_total)
        print(flush=True)
    # return

    from mpi4py import MPI
    MPI.COMM_WORLD.barrier()

    for j_total in range(mpi_rank, n_total, mpi_size):
        j_reduction = j_total % n_reductions
        j_rest = j_total // n_reductions

        j_power_of_bmag = j_rest % n_powers_of_bmag
        j_rest = j_rest // n_powers_of_bmag

        j_power_of_gds22 = j_rest % n_powers_of_gds22
        j_rest = j_rest // n_powers_of_gds22

        j_threshold = j_rest % n_thresholds
        j_rest = j_rest // n_thresholds

        j_gamma = j_rest % n_gammas
        j_rest = j_rest // n_gammas

        j_beta = j_rest % n_betas
        j_rest = j_rest // n_betas

        j_alpha = j_rest

        threshold = thresholds[j_threshold]
        alpha = alphas[j_alpha]
        beta = betas[j_beta]
        gamma = gammas[j_gamma]
        power_of_gds22 = powers_of_gds22[j_power_of_gds22]
        power_of_bmag = powers_of_bmag[j_power_of_bmag]

        x = cvdrift - threshold
        data = (
            (np.heaviside(x, 0) * (1 + gamma * x) + alpha * cvdrift + beta)
            * gds22**power_of_gds22
            * bmag**power_of_bmag
        )
        name = (
            f"[Heaviside(B⁻²𝗕×κ⋅∇y - {threshold:g}) + {gamma} ReLU(B⁻²𝗕×κ⋅∇y - {threshold:g}) + {alpha} B⁻²𝗕×κ⋅∇y + {beta}]"
            f" |∇x|²^{power_of_gds22}"
            f" B^{power_of_bmag}"
        )
        reduction, reduction_name = reductions_func(
            data, j_reduction
        )
        # In evaluator, set index = mpi_rank so the evaluator will always evaluate the cost function
        evaluator(reduction, f"{reduction_name}({name})", mpi_rank)

        if j_total % 1000 == 0:
            print(f"index: {j_total}  name: {name}", flush=True)


def compute_fn_20241211(
    data,
    mpi_rank, 
    mpi_size, 
    evaluator,
    reductions_func = reductions_20241108,
    n_B_powers=2,
):
    """
    This is a re-implementation of compute_fn_20241108 that should be faster.
    """
    z_functions = data["z_functions"]
    feature_tensor = data["feature_tensor"]
    scalars = data["scalars"]
    scalar_feature_matrix = data["scalar_feature_matrix"]
    z_functions = meaningful_names(z_functions)
    n_scalars = len(scalars)

    n_reductions = reductions_func(1, 1, return_n_reductions=True)
    print("n_reductions:", n_reductions)

    Bmag = feature_tensor[:, :, 0]

    # Add local shear as a feature:
    F, F_names = add_local_shear(feature_tensor, z_functions, include_integral=False)
    print("names after adding local shear:", F_names)

    # Add inverse quantities:
    inverse_tensor, inverse_names = make_inverse_quantities(F, F_names)
    F, F_names = combine_tensors(F, F_names, inverse_tensor, inverse_names)
    print("names after adding inverse quantities:", F_names)
    n_z_functions = len(F_names)
    del inverse_tensor  # Free up memory

    M, M_names = heaviside_transformations(F, F_names, long_names=True)
    identity_tensor = np.ones((F.shape[0], F.shape[1], 1))
    M, _ = combine_tensors(identity_tensor, [""], M, M_names)
    M_names = [""] + [n + " " for n in M_names]
    n_masks = len(M_names)

    # Tuples are (exponent, string)
    if n_B_powers == 1:
        extra_powers_of_B = [(0, "")]
    elif n_B_powers == 2:
        extra_powers_of_B = [(0, ""), (-1, " / B")]
    elif n_B_powers == 3:
        extra_powers_of_B = [(0, ""), (-1, " / B"), (-2, " / B²")]
    else:
        raise ValueError("Invalid n_B_powers")
    assert n_B_powers == len(extra_powers_of_B)
    
    # First set n_C to an upper bound, given by including X / X pairs that will
    # be dropped later:
    n_C = (n_z_functions * (n_z_functions + 1)) // 2 + n_z_functions
    j_combos_1_arr = np.zeros(n_C, dtype=np.int32)
    j_combos_2_arr = np.zeros(n_C, dtype=np.int32)
    index = 0
    for j_combos_1 in range(n_z_functions):
        for j_combos_2 in range(-1, j_combos_1 + 1):
            # # Don't multiply a feature by its own inverse:
            if j_combos_2 >= 0:
                name1 = F_names[j_combos_1]
                name2 = F_names[j_combos_2]
                if name1 == "1/" + name2 or name2 == "1/" + name1:
                    continue

            j_combos_1_arr[index] = j_combos_1
            j_combos_2_arr[index] = j_combos_2
            index += 1
    n_C = index  # Will be smaller than the original n_C because X / X was skipped above.

    n_total = n_C * n_reductions * n_B_powers * n_masks + n_scalars
        
    if mpi_rank == 0:
        print("names after adding local shear:", F_names)
        print("n_z_functions:", n_z_functions)
        print("n_C:", n_C)
        print("n_masks:", n_masks)
        print("n_reductions:", n_reductions)
        print("n_B_powers:", n_B_powers)
        print("n_scalars:", n_scalars)
        print(
            "Total number of features that will be evaluated:",
            n_total,
            flush=True,
        )
        print(flush=True)

    from mpi4py import MPI
    MPI.COMM_WORLD.barrier()

    index = 0
    # Try all the extra scalar features:
    n_scalars = len(scalars)
    for j in range(n_scalars):
        evaluator(scalar_feature_matrix[:, j], scalars[j], index)
        index += 1

    # Now the main features:
    index_for_evaluator = (
        mpi_rank  # so the evaluator will always evaluate the cost function
    )
    for j_total in range(mpi_rank, n_total, mpi_size):
        j_reduction = j_total % n_reductions
        j_rest = j_total // n_reductions

        j_power_of_bmag = j_rest % n_B_powers
        j_rest = j_rest // n_B_powers

        j_mask = j_rest % n_masks
        j_rest = j_rest // n_masks

        j_C = j_rest

        j_z_function1 = j_combos_1_arr[j_C]
        j_z_function2 = j_combos_2_arr[j_C]
        name1 = F_names[j_z_function1]
        if j_z_function2 < 0:
            C = F[:, :, j_z_function1]
            name = name1
        else:
            C = F[:, :, j_z_function1] * F[:, :, j_z_function2]
            name2 = F_names[j_z_function2]
            name = f"{name1} {name2}"

        # Don't include B / B:
        if j_power_of_bmag > 0 and (
            j_z_function1 == 0 or j_z_function2 == 0
        ):
            continue

        data = (
            M[:, :, j_mask] * C * Bmag**extra_powers_of_B[j_power_of_bmag][0]
        )
        z_function_name = f"{M_names[j_mask]}{name}{extra_powers_of_B[j_power_of_bmag][1]}"
        reduction, reduction_name = reductions_func(
            data, j_reduction
        )
        final_name = f"{reduction_name}({z_function_name})"
        evaluator(
            reduction, final_name, index_for_evaluator
        )
        if j_total % 1000 == 0:
            print(f"index: {j_total}  name: {final_name}", flush=True)

        index += 1


def create_test_features():
    raw_tensor, raw_names, Y = load_tensor("test")
    # n_data = 10, n_z = 96, n_quantities = 7
    n_quantities = 2
    raw_tensor = raw_tensor[:, :, :n_quantities]
    raw_names = raw_names[:n_quantities]
    print("raw_tensor.shape:", raw_tensor.shape)
    print("Y.shape:", Y.shape)

    tsfresh_custom_features = {
        "maximum": None,
        "minimum": None,
    }

    single_quantity_df = tensor_to_tsfresh_dataframe(raw_tensor, raw_names)
    features = extract_features(
        single_quantity_df,
        column_id="j_tube",
        column_sort="z",
        default_fc_parameters=tsfresh_custom_features,
    )
    features["Y"] = Y
    drop_special_characters_from_column_names(features)

    filename = "test_features.pkl"
    features.to_pickle(filename)


def recursive_feature_elimination(features_filename, n_features, step):
    output_filename = features_filename[:-4] + f"_RFE{n_features}.pkl"
    print("Results will be saved in", output_filename)

    data = read_pickle(features_filename)
    Y_all = data["Y"]
    X_all = data.drop(columns="Y")
    feature_names = X_all.columns

    estimator = make_pipeline(
        StandardScaler(), lgb.LGBMRegressor(importance_type="gain", force_col_wise=True)
    )

    rfe = RFE(
        estimator,
        step=step,
        n_features_to_select=n_features,
        verbose=3,
        importance_getter="named_steps.lgbmregressor.feature_importances_",
    )
    rfe.fit(X_all, Y_all)

    print("sum(rfe.ranking_ <= 1)", sum(rfe.ranking_ <= 1))
    print("sum(rfe.support_)", sum(rfe.support_))

    # Make a DataFrame with the selected features and save the result:
    X_subset = X_all.to_numpy()[:, rfe.support_]
    feature_names_subset = feature_names[rfe.support_]
    df = DataFrame(X_subset, columns=feature_names_subset)
    print("Size of the reduced feature set:", df.shape)
    df["Y"] = Y_all
    df.to_pickle(output_filename)
    print("Results are now saved in", output_filename)


def pick_features_from_SFS_results(sfs_file, features_file, n_features):
    """
    Given the results of a SequentialFeatureSelector, pick the best set of
    n_features features out of the features in features_file, and save the
    results in a new features file.
    """
    with open(sfs_file, "rb") as f:
        sfs = pickle.load(f)

    big_df = read_pickle(features_file)

    best_features = sfs.subsets_[n_features]["feature_names"]
    print("Keeping these features:\n")
    for f in best_features:
        print(f)

    small_df = big_df[list(best_features) + ["Y"]]
    output_filename = f"{features_file[:-4]}_SFS{n_features}.pkl"
    print("\nSaving results in", output_filename)
    small_df.to_pickle(output_filename)
