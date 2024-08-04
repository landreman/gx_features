import unittest
import numpy as np

from gx_features.calculations import differentiate, compute_mean_k_parallel


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

        features, new_names = compute_mean_k_parallel(raw_feature_tensor, names, include_argmax=True)
        assert new_names == ["foo__mean_kpar", "bar__mean_kpar", "foo__argmax_kpar", "bar__argmax_kpar"]

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

        features, new_names = compute_mean_k_parallel(raw_feature_tensor, names, include_argmax=True)

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
