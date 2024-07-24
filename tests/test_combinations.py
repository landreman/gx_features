import unittest
import numpy as np

from gx_features.io import load
from gx_features.combinations import add_local_shear
from gx_features.calculations import differentiate

class Tests(unittest.TestCase):
    def test_add_local_shear(self):
        # First check the case include_integral = True:

        data = load(True)
        feature_tensor = data["feature_tensor"]
        new_feature_tensor = add_local_shear(feature_tensor, include_integral=True)
        assert new_feature_tensor.shape == (data["n_data"], data["n_z"], 9)

        # z_functions: ['bmag', 'gbdrift', 'cvdrift', 'gbdrift0_over_shat', 'gds2', 'gds21_over_shat', 'gds22_over_shat_squared']
    
        np.testing.assert_allclose(
            new_feature_tensor[:, :, 7],
            differentiate(feature_tensor[:, :, 5] / feature_tensor[:, :, 6]),
        )

        np.testing.assert_allclose(
            new_feature_tensor[:, :, 8],
            feature_tensor[:, :, 5] / feature_tensor[:, :, 6],
        )

        # Now check the case include_integral = False:
        
        data = load(True)
        feature_tensor = data["feature_tensor"]
        new_feature_tensor = add_local_shear(feature_tensor, include_integral=False)
        assert new_feature_tensor.shape == (data["n_data"], data["n_z"], 8)

        np.testing.assert_allclose(
            new_feature_tensor[:, :, 7],
            differentiate(feature_tensor[:, :, 5] / feature_tensor[:, :, 6]),
        )
