import unittest
import numpy as np
from tsfresh import extract_features

from gx_features.calculations import (
    differentiate,
    compute_mean_k_parallel,
    compute_longest_nonzero_interval,
    compute_mask_for_longest_true_interval,
    compute_reductions,
)
from gx_features.io import load_tensor
from gx_features.utils import tensor_to_tsfresh_dataframe, simplify_names


class Tests(unittest.TestCase):
    def test_differentiate(self):
        n_data = 7
        n_quantities = 2
        n_z = 96
        z = np.linspace(-np.pi, np.pi, n_z, endpoint=False)

        raw_feature_tensor = np.zeros((n_data, n_z, n_quantities))
        for j_data in range(n_data):
            raw_feature_tensor[j_data, :, 0] = 3 * np.sin(j_data * z + 1)

        raw_feature_tensor[:, :, 1] = differentiate(raw_feature_tensor[:, :, 0])

        for j_data in range(n_data):
            np.testing.assert_allclose(
                raw_feature_tensor[j_data, :, 1],
                3 * j_data * np.cos(j_data * z + 1),
                rtol=3e-2,
            )

    def test_mean_k_parallel(self):
        # First try a pure frequency:

        n_data = 4
        n_quantities = 2
        n_z = 96
        # n_z = 16
        z = np.linspace(-np.pi, np.pi, n_z, endpoint=False)
        names = ["foo", "bar"]

        raw_feature_tensor = np.zeros((n_data, n_z, n_quantities))
        for j_data in range(n_data):
            raw_feature_tensor[j_data, :, 0] = 3 * np.sin((j_data + 1) * z + 1)
            raw_feature_tensor[j_data, :, 1] = 17 * np.cos((j_data + 1) * z - 1)

        features, new_names = compute_mean_k_parallel(raw_feature_tensor, names)
        assert new_names == ["foo__mean_kpar", "bar__mean_kpar"]

        for j_data in range(n_data):
            np.testing.assert_allclose(
                features[j_data, :],
                j_data + 1,
                rtol=1e-13,
            )

        # Now try a sum of frequencies:

        for j_data in range(n_data):
            raw_feature_tensor[j_data, :, 0] = 3 * (
                np.sin((j_data + 1) * z + 1) + np.sin((j_data + 2) * z + j_data)
            )
            raw_feature_tensor[j_data, :, 1] = 17 * (
                np.cos((j_data + 1) * z - 1) + np.cos((j_data + 2) * z - j_data)
            )

        features, _ = compute_mean_k_parallel(raw_feature_tensor, names)

        for j_data in range(n_data):
            np.testing.assert_allclose(
                features[j_data, :],
                j_data + 1.5,
                rtol=1e-13,
            )

    def test_mean_k_parallel_with_argmax(self):
        # First try a pure frequency:

        n_data = 4
        n_quantities = 2
        n_z = 96
        # n_z = 16
        z = np.linspace(-np.pi, np.pi, n_z, endpoint=False)
        names = ["foo", "bar"]

        raw_feature_tensor = np.zeros((n_data, n_z, n_quantities))
        for j_data in range(n_data):
            raw_feature_tensor[j_data, :, 0] = 3 * np.sin((j_data + 1) * z + 1)
            raw_feature_tensor[j_data, :, 1] = 17 * np.cos((j_data + 2) * z - 1)

        features, new_names = compute_mean_k_parallel(
            raw_feature_tensor, names, include_argmax=True
        )
        assert new_names == [
            "foo__mean_kpar",
            "bar__mean_kpar",
            "foo__argmax_kpar",
            "bar__argmax_kpar",
        ]

        for j_data in range(n_data):
            np.testing.assert_allclose(
                features[j_data, 0],
                j_data + 1,
                rtol=1e-13,
            )
            np.testing.assert_allclose(
                features[j_data, 1],
                j_data + 2,
                rtol=1e-13,
            )
            np.testing.assert_allclose(
                features[j_data, 2],
                j_data + 1,
                rtol=1e-13,
            )
            np.testing.assert_allclose(
                features[j_data, 3],
                j_data + 2,
                rtol=1e-13,
            )

        # Now try a sum of frequencies:

        for j_data in range(n_data):
            raw_feature_tensor[j_data, :, 0] = 3 * (
                2 * np.sin((j_data + 1) * z + 1) + np.cos((j_data + 2) * z + j_data)
            )
            raw_feature_tensor[j_data, :, 1] = 17 * (
                np.cos((j_data + 1) * z - 1) + 3 * np.sin((j_data + 2) * z - j_data)
            )

        features, new_names = compute_mean_k_parallel(
            raw_feature_tensor, names, include_argmax=True
        )

        for j_data in range(n_data):
            np.testing.assert_allclose(
                features[j_data, 0],
                j_data + 1 + (1.0 / 3.0),
                rtol=1e-13,
            )
            np.testing.assert_allclose(
                features[j_data, 1],
                j_data + 1.75,
                rtol=1e-13,
            )
            np.testing.assert_allclose(
                features[j_data, 2],
                j_data + 1,
                rtol=1e-13,
            )
            np.testing.assert_allclose(
                features[j_data, 3],
                j_data + 2,
                rtol=1e-13,
            )

    def test_mean_k_parallel_with_0s(self):
        raw_features = np.random.rand(10, 12, 3)
        # Set a few rows to constants
        raw_features[2, :, 1] = 0
        raw_features[8, :, 1] = 0
        raw_features[5, :, 2] = 1
        raw_features[0, :, 0] = -2
        names = ["foo", "bar", "yaz"]
        for include_argmax in [False, True]:
            features, new_names = compute_mean_k_parallel(
                raw_features,
                names,
                include_argmax=include_argmax,
            )

    def test_compute_longest_nonzero_interval(self):
        n_z = 8
        n_data = n_z + 1
        n_quantities = 6
        raw_features = np.zeros((n_data, n_z, n_quantities))
        raw_features[0, :, 0] = 4 * np.ones(n_z)
        raw_features[0, :, 2] = [2, 1, -3, 0, 0, 0, 9, 0]
        raw_features[0, :, 3] = [0, -1, 1, 0, 0, -4, 8, 0]
        raw_features[0, :, 4] = [0, 8, 0, 0, 3, 0, 1, -1]
        raw_features[0, :, 5] = [0, -3, 1, 1, -1, 1, -2, 1]
        for j in range(1, n_data):
            raw_features[j, :, :] = np.roll(raw_features[0, :, :], j, axis=0)
        names = ["foo", "bar", "baz", "qux", "zim", "fap"]

        features, new_names = compute_longest_nonzero_interval(raw_features, names)
        assert features.shape == (n_data, n_quantities - 2)
        np.testing.assert_allclose(features[:, 0], 3)
        np.testing.assert_allclose(features[:, 1], 2)
        np.testing.assert_allclose(features[:, 2], 2)
        np.testing.assert_allclose(features[:, 3], 7)

    def test_compute_mask_for_longest_true_interval(self):
        n_z = 8
        n_data = n_z + 1
        n_quantities = 6
        raw_features = np.zeros((n_data, n_z, n_quantities))
        features_should_be = np.zeros((n_data, n_z, n_quantities))
        raw_features[0, :, 0] = np.full(n_z, True)
        raw_features[0, :, 2] = [1, 1, 1, 0, 0, 0, 1, 0]
        raw_features[0, :, 3] = [1, 1, 0, 1, 0, 0, 1, 1]
        raw_features[0, :, 4] = [0, 1, 0, 0, 1, 0, 1, 1]
        raw_features[0, :, 5] = [0, 1, 1, 1, 1, 1, 1, 1]

        features_should_be[0, :, 0] = np.ones(n_z)
        features_should_be[0, :, 2] = [1, 1, 1, 0, 0, 0, 0, 0]
        features_should_be[0, :, 3] = [1, 1, 0, 0, 0, 0, 1, 1]
        features_should_be[0, :, 4] = [0, 0, 0, 0, 0, 0, 1, 1]
        features_should_be[0, :, 5] = [0, 1, 1, 1, 1, 1, 1, 1]

        for j in range(1, n_data):
            raw_features[j, :, :] = np.roll(raw_features[0, :, :], j, axis=0)
            features_should_be[j, :, :] = np.roll(
                features_should_be[0, :, :], j, axis=0
            )
        names = ["foo", "bar", "baz", "qux", "zim", "fap"]

        mask = compute_mask_for_longest_true_interval(raw_features)
        assert mask.shape == (n_data, n_z, n_quantities)
        np.testing.assert_allclose(mask, features_should_be, atol=1e-14)
        # for j_data in range(n_data):
        #     for j_quantity in range(n_quantities):
        #         print("j_data:", j_data, "j_quantity:", j_quantity)
        #         np.testing.assert_allclose(mask[j_data, :, j_quantity], features_should_be[j_data, :, j_quantity])

    def test_compute_reductions(self):
        """My reduction functions should be equivalent to the tsfresh ones."""
        tensor, names, Y = load_tensor("test")
        n_data, n_z, n_quantities = tensor.shape
        names = simplify_names(names)
        df1 = compute_reductions(
            tensor,
            names,
            max=True,
            min=True,
            mean=True,
            median=True,
            rms=True,
            variance=True,
            skewness=True,
            quantiles=[0.1, 0.7],
            count_above=[0, 1.2],
            fft_coefficients=[2, 3, 4],
        )
        print(df1)

        tsfresh_feature_options = {
            "maximum": None,
            "minimum": None,
            "mean": None,
            "median": None,
            "root_mean_square": None,
            "variance": None,
            "skewness": None,
            "quantile": [{"q": 0.1}, {"q": 0.7}],
            "count_above": [{"t": 0}, {"t": 1.2}],
            "fft_coefficient": [
                {"attr": "abs", "coeff": 2},
                {"attr": "abs", "coeff": 3},
                {"attr": "abs", "coeff": 4},
            ],
        }

        df_for_tsfresh = tensor_to_tsfresh_dataframe(tensor, names)
        df2 = extract_features(
            df_for_tsfresh,
            column_id="j_tube",
            column_sort="z",
            default_fc_parameters=tsfresh_feature_options,
        )
        print(df2)

        array1 = df1.to_numpy()
        array2 = df2.to_numpy()
        assert array1.shape == array2.shape
        n_features_per_quantity = int(array1.shape[1] / n_quantities)
        array2_rearranged = np.zeros_like(array1)
        for j_quantity in range(n_quantities):
            array2_rearranged[:, j_quantity::n_quantities] = array2[
                :,
                j_quantity
                * n_features_per_quantity : (j_quantity + 1)
                * n_features_per_quantity,
            ]

        # np.set_printoptions(linewidth=400)
        # print("array1:")
        # print(array1)
        # print("array2_rearranged:")
        # print(array2_rearranged)
        np.testing.assert_allclose(array1, array2_rearranged, atol=1e-13)
