import unittest
import numpy as np

from gx_features.calculations import differentiate

class Tests(unittest.TestCase):
    def test_differentiate(self):
        n_data = 7
        n_quantities = 2
        n_z = 96
        z = np.linspace(-np.pi, np.pi, n_z, endpoint=False)

        feature_tensor = np.zeros((n_data, n_z, n_quantities))
        for j_data in range(n_data):
            feature_tensor[j_data, :, 0] = 3 * np.sin(j_data * z + 1)

        feature_tensor[:, :, 1] = differentiate(feature_tensor[:, :, 0])

        for j_data in range(n_data):
            np.testing.assert_allclose(
                feature_tensor[j_data, :, 1],
                3 * j_data * np.cos(j_data * z + 1),
                rtol=3e-2,
            )
