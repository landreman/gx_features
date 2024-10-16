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
from tsfresh import extract_features
import lightgbm as lgb
from memory_profiler import profile

from .io import load_all, load_tensor
from .calculations import (
    compute_mean_k_parallel,
    compute_max_minus_min,
    compute_reductions,
    compute_mask_for_longest_true_interval,
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
from .utils import (
    tensor_to_tsfresh_dataframe,
    drop_nearly_constant_features,
    drop_special_characters_from_column_names,
    simplify_names,
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
def create_features_20241011_01(n_data=None, mpi=False):
    """
    Same as 20240804_01, but for finite-beta rather than vacuum, so cvdrift is
    also included. Also, the scalar features [nfp, iota, shat, d_pressure_d_s]
    are added.

    If n_data is None, all data entries will be used.
    If n_data is an integer, the data will be trimmed to the first n_data entries.
    """
    start_time = time.time()
    data = load_all("20241005")
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
        raw_tensor, extra_scalar_features = distribute_work_mpi(raw_tensor, extra_scalar_features)
        from mpi4py import MPI
        proc0 = (MPI.COMM_WORLD.rank == 0)
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
    print(time.time() - start_time, "Done combining inverse quantities with original raw features", flush=True)

    CF, CF_names = make_feature_product_combinations(F, F_names)
    print(time.time() - start_time, "Done forming feature product combinations", flush=True)

    M, M_names = heaviside_transformations(F, F_names)
    proc0_print(time.time() - start_time, "Done creating Heaviside transformations", flush=True)
    proc0_print("M_names:", M_names, flush=True)

    MF, MF_names = make_pairwise_products_from_2_sets(F, F_names, M, M_names)
    print(time.time() - start_time, "Done creating MF features", flush=True)
    MCF, MCF_names = make_pairwise_products_from_2_sets(CF, CF_names, M, M_names)
    print(time.time() - start_time, "Done creating MCF features", flush=True)

    tensor_before_inv_bmag, names_before_inv_bmag = combine_tensors(
        F, F_names, MF, MF_names, CF, CF_names, MCF, MCF_names
    )
    del F, MF, CF, MCF  # Free up memory
    print(time.time() - start_time, "Done combining tensors before dividing by bmag", flush=True)

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
    print(time.time() - start_time, "Done combining tensors with and without 1/B", flush=True)

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
        flush=True
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

        filename = "20241005-01_features_20241011_01"
        features.to_pickle(filename + ".pkl")

        return features


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
