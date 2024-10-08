import unittest
import numpy as np

from gx_features.io import load_all, load_tensor
from gx_features.combinations import (
    add_local_shear,
    remove_cvdrift,
    create_masks,
    make_pairwise_products_from_2_sets,
    make_feature_product_combinations,
    make_feature_quotient_combinations,
    make_inverse_quantities,
    make_feature_product_and_quotient_combinations,
    heaviside_transformations,
)
from gx_features.calculations import differentiate


class Tests(unittest.TestCase):
    def test_remove_cvdrift(self):
        data = load_all("test")
        feature_tensor = data["feature_tensor"]
        new_feature_tensor, new_names = remove_cvdrift(
            feature_tensor, data["z_functions"]
        )
        assert new_feature_tensor.shape == (data["n_data"], data["n_z"], 6)
        np.testing.assert_allclose(
            feature_tensor[:, :, :2], new_feature_tensor[:, :, :2]
        )
        np.testing.assert_allclose(
            feature_tensor[:, :, 3:], new_feature_tensor[:, :, 2:]
        )
        assert new_names == [
            "bmag",
            "gbdrift",
            "gbdrift0_over_shat",
            "gds2",
            "gds21_over_shat",
            "gds22_over_shat_squared",
        ]

    def test_add_local_shear(self):
        # First check the case include_integral = True:

        data = load_all("test")
        feature_tensor = data["feature_tensor"]
        new_feature_tensor, _ = add_local_shear(
            feature_tensor, data["z_functions"], include_integral=True
        )
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

        feature_tensor, names, Y = load_tensor("test")
        new_feature_tensor, _ = add_local_shear(
            feature_tensor, names, include_integral=False
        )
        assert new_feature_tensor.shape == (data["n_data"], data["n_z"], 8)

        np.testing.assert_allclose(
            new_feature_tensor[:, :, 7],
            differentiate(feature_tensor[:, :, 5] / feature_tensor[:, :, 6]),
        )

        # Now repeat, after removing cvdrift:

        data = load_all("test")
        feature_tensor, names = remove_cvdrift(
            data["feature_tensor"], data["z_functions"]
        )
        new_feature_tensor, _ = add_local_shear(
            feature_tensor, data["z_functions"], include_integral=True
        )
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

        feature_tensor, names, Y = load_tensor("test")
        feature_tensor, names = remove_cvdrift(feature_tensor, names)
        new_feature_tensor, _ = add_local_shear(
            feature_tensor, names, include_integral=False
        )
        assert new_feature_tensor.shape == (data["n_data"], data["n_z"], 7)

        np.testing.assert_allclose(
            new_feature_tensor[:, :, 6],
            differentiate(feature_tensor[:, :, 4] / feature_tensor[:, :, 5]),
        )

    def test_create_masks(self):
        data = load_all("test")
        feature_tensor = data["feature_tensor"]
        masks, mask_names = create_masks(feature_tensor)
        assert masks.shape[2] == 5

        gbdrift_index = data["z_functions"].index("gbdrift")
        cvdrift_index = data["z_functions"].index("cvdrift")
        assert cvdrift_index > gbdrift_index
        np.testing.assert_allclose(masks[:, :, 0], 1)
        np.testing.assert_allclose(
            masks[:, :, 1], feature_tensor[:, :, gbdrift_index] >= 0
        )
        np.testing.assert_allclose(
            masks[:, :, 2], feature_tensor[:, :, gbdrift_index] <= 0
        )
        np.testing.assert_allclose(
            masks[:, :, 3], feature_tensor[:, :, cvdrift_index] >= 0
        )
        np.testing.assert_allclose(
            masks[:, :, 4], feature_tensor[:, :, cvdrift_index] <= 0
        )

        # Now repeat after removing cvdrift:

        feature_tensor, names = remove_cvdrift(
            data["feature_tensor"], data["z_functions"]
        )
        masks, mask_names = create_masks(feature_tensor)
        assert masks.shape[2] == 3
        np.testing.assert_allclose(masks[:, :, 0], 1)
        np.testing.assert_allclose(
            masks[:, :, 1], feature_tensor[:, :, gbdrift_index] >= 0
        )
        np.testing.assert_allclose(
            masks[:, :, 2], feature_tensor[:, :, gbdrift_index] <= 0
        )

    def test_feature_mask_combinations(self):
        feature_tensor, names, Y = load_tensor("test")
        masks, mask_names = create_masks(feature_tensor)
        combinations, names = make_pairwise_products_from_2_sets(
            feature_tensor, names, masks, mask_names
        )
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

        names_should_be = [
            "bmag",
            "gbdrift",
            "cvdrift",
            "gbdrift0_over_shat",
            "gds2",
            "gds21_over_shat",
            "gds22_over_shat_squared",
            "bmag_gbdriftPos",
            "gbdrift_gbdriftPos",
            "cvdrift_gbdriftPos",
            "gbdrift0_over_shat_gbdriftPos",
            "gds2_gbdriftPos",
            "gds21_over_shat_gbdriftPos",
            "gds22_over_shat_squared_gbdriftPos",
            "bmag_gbdriftNeg",
            "gbdrift_gbdriftNeg",
            "cvdrift_gbdriftNeg",
            "gbdrift0_over_shat_gbdriftNeg",
            "gds2_gbdriftNeg",
            "gds21_over_shat_gbdriftNeg",
            "gds22_over_shat_squared_gbdriftNeg",
            "bmag_cvdriftPos",
            "gbdrift_cvdriftPos",
            "cvdrift_cvdriftPos",
            "gbdrift0_over_shat_cvdriftPos",
            "gds2_cvdriftPos",
            "gds21_over_shat_cvdriftPos",
            "gds22_over_shat_squared_cvdriftPos",
            "bmag_cvdriftNeg",
            "gbdrift_cvdriftNeg",
            "cvdrift_cvdriftNeg",
            "gbdrift0_over_shat_cvdriftNeg",
            "gds2_cvdriftNeg",
            "gds21_over_shat_cvdriftNeg",
            "gds22_over_shat_squared_cvdriftNeg",
        ]
        assert names == names_should_be

    def test_make_feature_product_combinations(self):
        feature_tensor, names, Y = load_tensor("test")
        n_quantities = 3
        feature_tensor = feature_tensor[:, :, :n_quantities]
        names = names[:n_quantities]
        product_features, product_names = make_feature_product_combinations(
            feature_tensor, names
        )
        assert product_names == [
            "bmag_x_gbdrift",
            "bmag_x_cvdrift",
            "gbdrift_x_cvdrift",
        ]
        n_data, n_z, _ = feature_tensor.shape
        assert product_features.shape == (n_data, n_z, 3)
        np.testing.assert_allclose(
            product_features[:, :, 0], feature_tensor[:, :, 0] * feature_tensor[:, :, 1]
        )
        np.testing.assert_allclose(
            product_features[:, :, 1], feature_tensor[:, :, 0] * feature_tensor[:, :, 2]
        )
        np.testing.assert_allclose(
            product_features[:, :, 2], feature_tensor[:, :, 1] * feature_tensor[:, :, 2]
        )

    def test_make_feature_quotient_combinations(self):
        # First try the first 3 features:

        feature_tensor, names, Y = load_tensor("test")
        n_quantities = 3
        feature_tensor = feature_tensor[:, :, :n_quantities]
        names = names[:n_quantities]
        quotient_tensor, quotient_names = make_feature_quotient_combinations(
            feature_tensor, names
        )
        assert quotient_names == ["gbdrift_/_bmag", "cvdrift_/_bmag"]
        n_data, n_z, _ = feature_tensor.shape
        assert quotient_tensor.shape == (n_data, n_z, 2)
        assert len(quotient_names) == 2
        np.testing.assert_allclose(
            quotient_tensor[:, :, 0], feature_tensor[:, :, 1] / feature_tensor[:, :, 0]
        )
        np.testing.assert_allclose(
            quotient_tensor[:, :, 1], feature_tensor[:, :, 2] / feature_tensor[:, :, 0]
        )

        # Now try the first 4 features, after removing cvdrift:

        feature_tensor, names, Y = load_tensor("test")
        feature_tensor, names = remove_cvdrift(feature_tensor, names)
        n_quantities = 4
        feature_tensor = feature_tensor[:, :, :n_quantities]
        names = names[:n_quantities]
        quotient_tensor, quotient_names = make_feature_quotient_combinations(
            feature_tensor, names
        )
        assert quotient_names == [
            "gbdrift_/_bmag",
            "gbdrift0_over_shat_/_bmag",
            "gds2_/_bmag",
            "bmag_/_gds2",
            "gbdrift_/_gds2",
            "gbdrift0_over_shat_/_gds2",
        ]
        n_data, n_z, _ = feature_tensor.shape
        assert quotient_tensor.shape == (n_data, n_z, 6)
        assert len(quotient_names) == 6
        np.testing.assert_allclose(
            quotient_tensor[:, :, 0], feature_tensor[:, :, 1] / feature_tensor[:, :, 0]
        )
        np.testing.assert_allclose(
            quotient_tensor[:, :, 1], feature_tensor[:, :, 2] / feature_tensor[:, :, 0]
        )
        np.testing.assert_allclose(
            quotient_tensor[:, :, 2], feature_tensor[:, :, 3] / feature_tensor[:, :, 0]
        )
        np.testing.assert_allclose(
            quotient_tensor[:, :, 3], feature_tensor[:, :, 0] / feature_tensor[:, :, 3]
        )
        np.testing.assert_allclose(
            quotient_tensor[:, :, 4], feature_tensor[:, :, 1] / feature_tensor[:, :, 3]
        )
        np.testing.assert_allclose(
            quotient_tensor[:, :, 5], feature_tensor[:, :, 2] / feature_tensor[:, :, 3]
        )

    def test_make_inverse_quantities(self):
        # First try the first 3 features:

        feature_tensor, names, Y = load_tensor("test")
        n_quantities = 3
        feature_tensor = feature_tensor[:, :, :n_quantities]
        names = names[:n_quantities]
        inverse_tensor, inverse_names = make_inverse_quantities(feature_tensor, names)
        assert inverse_names == ["1/bmag"]
        n_data, n_z, _ = feature_tensor.shape
        assert inverse_tensor.shape == (n_data, n_z, 1)
        np.testing.assert_allclose(inverse_tensor[:, :, 0], 1 / feature_tensor[:, :, 0])

        # Now try all raw quantities:

        feature_tensor, names, Y = load_tensor("test")
        inverse_tensor, inverse_names = make_inverse_quantities(feature_tensor, names)
        assert inverse_names == ["1/bmag", "1/gds2", "1/gds22_over_shat_squared"]
        n_data, n_z, _ = feature_tensor.shape
        assert inverse_tensor.shape == (n_data, n_z, 3)
        np.testing.assert_allclose(inverse_tensor[:, :, 0], 1 / feature_tensor[:, :, 0])
        np.testing.assert_allclose(inverse_tensor[:, :, 1], 1 / feature_tensor[:, :, 4])
        np.testing.assert_allclose(inverse_tensor[:, :, 2], 1 / feature_tensor[:, :, 6])

    def test_make_feature_products_with_inverses(self):
        """Make sure a quantity is not multiplied by its own inverse"""
        feature_tensor, names, Y = load_tensor("test")
        n_quantities = 2
        feature_tensor = feature_tensor[:, :, :n_quantities]
        names = names[:n_quantities]
        inverse_tensor, inverse_names = make_inverse_quantities(feature_tensor, names)

        tensor = np.concatenate((feature_tensor, inverse_tensor), axis=2)
        names = names + inverse_names
        print("names:", names)
        product_features, product_names = make_feature_product_combinations(
            tensor, names
        )
        assert product_names == [
            "bmag_x_gbdrift",
            "gbdrift_x_1/bmag",
        ]
        n_data, n_z, _ = feature_tensor.shape
        assert product_features.shape == (n_data, n_z, 2)
        np.testing.assert_allclose(
            product_features[:, :, 0], feature_tensor[:, :, 0] * feature_tensor[:, :, 1]
        )
        np.testing.assert_allclose(
            product_features[:, :, 1], feature_tensor[:, :, 1] / feature_tensor[:, :, 0]
        )

    def test_make_feature_product_and_quotient_combinations(self):
        # First try the first 3 features:

        feature_tensor, names, Y = load_tensor("test")
        n_quantities = 3
        feature_tensor = feature_tensor[:, :, :n_quantities]
        names = names[:n_quantities]
        combinations_tensor, combinations_names = (
            make_feature_product_and_quotient_combinations(feature_tensor, names)
        )
        assert combinations_names == [
            "bmag_x_gbdrift",
            "bmag_x_cvdrift",
            "gbdrift_x_cvdrift",
            "gbdrift_/_bmag",
            "cvdrift_/_bmag",
        ]
        n_data, n_z, _ = feature_tensor.shape
        assert combinations_tensor.shape == (n_data, n_z, 5)
        assert len(combinations_names) == 5
        np.testing.assert_allclose(
            combinations_tensor[:, :, 0],
            feature_tensor[:, :, 1] * feature_tensor[:, :, 0],
        )
        np.testing.assert_allclose(
            combinations_tensor[:, :, 1],
            feature_tensor[:, :, 2] * feature_tensor[:, :, 0],
        )
        np.testing.assert_allclose(
            combinations_tensor[:, :, 2],
            feature_tensor[:, :, 2] * feature_tensor[:, :, 1],
        )
        np.testing.assert_allclose(
            combinations_tensor[:, :, 3],
            feature_tensor[:, :, 1] / feature_tensor[:, :, 0],
        )
        np.testing.assert_allclose(
            combinations_tensor[:, :, 4],
            feature_tensor[:, :, 2] / feature_tensor[:, :, 0],
        )

    def test_heaviside_transformations(self):
        feature_tensor, names, Y = load_tensor("test")
        n_data, n_z, n_quantities = feature_tensor.shape
        transformed_features, transformed_names = heaviside_transformations(
            feature_tensor, names
        )
        assert transformed_features.shape == (n_data, n_z, 8)
        assert transformed_names == [
            "gbdriftPos",
            "gbdriftNeg",
            "cvdriftPos",
            "cvdriftNeg",
            "gbdrift0_over_shatPos",
            "gbdrift0_over_shatNeg",
            "gds21_over_shatPos",
            "gds21_over_shatNeg",
        ]
        np.testing.assert_allclose(
            transformed_features[:, :, 0],
            np.where(feature_tensor[:, :, 1] > 0, 1, 0),
            atol=1e-14,
        )
        np.testing.assert_allclose(
            transformed_features[:, :, 1],
            np.where(feature_tensor[:, :, 1] < 0, 1, 0),
            atol=1e-14,
        )
        np.testing.assert_allclose(
            transformed_features[:, :, 2],
            np.where(feature_tensor[:, :, 2] > 0, 1, 0),
            atol=1e-14,
        )
        np.testing.assert_allclose(
            transformed_features[:, :, 3],
            np.where(feature_tensor[:, :, 2] < 0, 1, 0),
            atol=1e-14,
        )
        np.testing.assert_allclose(
            transformed_features[:, :, 4],
            np.where(feature_tensor[:, :, 3] > 0, 1, 0),
            atol=1e-14,
        )
        np.testing.assert_allclose(
            transformed_features[:, :, 5],
            np.where(feature_tensor[:, :, 3] < 0, 1, 0),
            atol=1e-14,
        )
        np.testing.assert_allclose(
            transformed_features[:, :, 6],
            np.where(feature_tensor[:, :, 5] > 0, 1, 0),
            atol=1e-14,
        )
        np.testing.assert_allclose(
            transformed_features[:, :, 7],
            np.where(feature_tensor[:, :, 5] < 0, 1, 0),
            atol=1e-14,
        )
