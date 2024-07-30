import numpy as np
from pandas import DataFrame
from tsfresh import extract_features

from .io import load_tensor
from .calculations import compute_mean_k_parallel
from .combinations import (
    add_local_shear,
    create_masks,
    make_feature_mask_combinations,
    make_feature_product_combinations,
    combine_tensors,
)
from .utils import tensor_to_tsfresh_dataframe, drop_nearly_constant_features


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
    feature_mask_tensor, feature_mask_names = make_feature_mask_combinations(
        single_quantity_tensor, single_quantity_names, masks, mask_names
    )

    # Create feature-pair-mask combinations:
    feature_pair_mask_tensor, feature_pair_mask_names = make_feature_mask_combinations(
        feature_product_tensor, feature_product_names, masks, mask_names
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

    if dataset == "test":
        filename += "20240601-01-kpar_and_pair_mask_features_test"
    elif dataset == "20240601":
        filename = "20240601-01-kpar_and_pair_mask_features"
    elif dataset == "20240726":
        filename = "20240726-01-kpar_and_pair_mask_features"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    features.to_pickle(filename + ".pkl")
