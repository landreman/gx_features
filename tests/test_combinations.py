import unittest
import numpy as np

from gx_features.io import load
from gx_features.combinations import add_local_shear, remove_cvdrift, create_masks, make_feature_mask_combinations
from gx_features.calculations import differentiate

class Tests(unittest.TestCase):
    def test_remove_cvdrift(self):
        data = load(True)
        feature_tensor = data["feature_tensor"]
        new_feature_tensor = remove_cvdrift(feature_tensor)
        assert new_feature_tensor.shape == (data["n_data"], data["n_z"], 6)
        np.testing.assert_allclose(feature_tensor[:, :, :2], new_feature_tensor[:, :, :2])
        np.testing.assert_allclose(feature_tensor[:, :, 3:], new_feature_tensor[:, :, 2:])
        
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

        # Now repeat, after removing cvdrift:

        data = load(True)
        feature_tensor = remove_cvdrift(data["feature_tensor"])
        new_feature_tensor = add_local_shear(feature_tensor, include_integral=True)
        assert new_feature_tensor.shape == (data["n_data"], data["n_z"], 8)

        # z_functions: ['bmag', 'gbdrift', 'cvdrift', 'gbdrift0_over_shat', 'gds2', 'gds21_over_shat', 'gds22_over_shat_squared']
    
        np.testing.assert_allclose(
            new_feature_tensor[:, :, 6],
            differentiate(feature_tensor[:, :, 4] / feature_tensor[:, :, 5]),
        )

        np.testing.assert_allclose(
            new_feature_tensor[:, :, 7],
            feature_tensor[:, :, 4] / feature_tensor[:, :, 5],
        )

        # Now check the case include_integral = False:
        
        data = load(True)
        feature_tensor = remove_cvdrift(data["feature_tensor"])
        new_feature_tensor = add_local_shear(feature_tensor, include_integral=False)
        assert new_feature_tensor.shape == (data["n_data"], data["n_z"], 7)

        np.testing.assert_allclose(
            new_feature_tensor[:, :, 6],
            differentiate(feature_tensor[:, :, 4] / feature_tensor[:, :, 5]),
        )

    def test_create_masks(self):
        data = load(True)
        feature_tensor = data["feature_tensor"]
        masks, mask_names = create_masks(feature_tensor)
        assert masks.shape[2] == 5

        gbdrift_index = data["z_functions"].index("gbdrift")
        cvdrift_index = data["z_functions"].index("cvdrift")
        assert cvdrift_index > gbdrift_index
        np.testing.assert_allclose(masks[:, :, 0], 1)
        np.testing.assert_allclose(masks[:, :, 1], feature_tensor[:, :, gbdrift_index] >= 0)
        np.testing.assert_allclose(masks[:, :, 2], feature_tensor[:, :, gbdrift_index] <= 0)
        np.testing.assert_allclose(masks[:, :, 3], feature_tensor[:, :, cvdrift_index] >= 0)
        np.testing.assert_allclose(masks[:, :, 4], feature_tensor[:, :, cvdrift_index] <= 0)

        # Now repeat after removing cvdrift:

        feature_tensor = remove_cvdrift(data["feature_tensor"])
        masks, mask_names = create_masks(feature_tensor)
        assert masks.shape[2] == 3
        np.testing.assert_allclose(masks[:, :, 0], 1)
        np.testing.assert_allclose(masks[:, :, 1], feature_tensor[:, :, gbdrift_index] >= 0)
        np.testing.assert_allclose(masks[:, :, 2], feature_tensor[:, :, gbdrift_index] <= 0)

    def test_feature_mask_combinations(self):
        data = load(True)
        feature_tensor = data["feature_tensor"]
        masks, mask_names = create_masks(feature_tensor)
        combinations, names = make_feature_mask_combinations(feature_tensor, data["z_functions"], masks, mask_names)
        print(names)
        n_masks = 5
        n_quantities = 7
        assert combinations.shape[2] == n_quantities * n_masks
        for j_quantity in range(n_quantities):
            for j_mask in range(n_masks):
                np.testing.assert_allclose(
                    combinations[:, :, j_mask * n_quantities + j_quantity],
                    feature_tensor[:, :, j_quantity] * masks[:, :, j_mask],
                )

        names_should_be = ['bmag', 'gbdrift', 'cvdrift', 'gbdrift0_over_shat', 'gds2', 'gds21_over_shat', 'gds22_over_shat_squared', 'bmag_gbdriftPos', 'gbdrift_gbdriftPos', 'cvdrift_gbdriftPos', 'gbdrift0_over_shat_gbdriftPos', 'gds2_gbdriftPos', 'gds21_over_shat_gbdriftPos', 'gds22_over_shat_squared_gbdriftPos', 'bmag_gbdriftNeg', 'gbdrift_gbdriftNeg', 'cvdrift_gbdriftNeg', 'gbdrift0_over_shat_gbdriftNeg', 'gds2_gbdriftNeg', 'gds21_over_shat_gbdriftNeg', 'gds22_over_shat_squared_gbdriftNeg', 'bmag_cvdriftPos', 'gbdrift_cvdriftPos', 'cvdrift_cvdriftPos', 'gbdrift0_over_shat_cvdriftPos', 'gds2_cvdriftPos', 'gds21_over_shat_cvdriftPos', 'gds22_over_shat_squared_cvdriftPos', 'bmag_cvdriftNeg', 'gbdrift_cvdriftNeg', 'cvdrift_cvdriftNeg', 'gbdrift0_over_shat_cvdriftNeg', 'gds2_cvdriftNeg', 'gds21_over_shat_cvdriftNeg', 'gds22_over_shat_squared_cvdriftNeg']
        assert names == names_should_be