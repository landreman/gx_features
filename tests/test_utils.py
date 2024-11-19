import unittest
import numpy as np
from tsfresh import extract_features

from gx_features.io import load_tensor
from gx_features.utils import (
    tensor_to_tsfresh_dataframe,
    make_test_dataframe,
    drop_nearly_constant_features,
    drop_special_characters_from_column_names,
    meaningful_names,
)


class Tests(unittest.TestCase):
    def test_tensor_to_tsfresh_dataframe(self):
        feature_tensor, names, Y = load_tensor("test")
        n_quantities = 2
        feature_tensor = feature_tensor[:, :, :n_quantities]
        names = names[:n_quantities]
        df = tensor_to_tsfresh_dataframe(feature_tensor, names)
        print(df)

        n_data, n_z, n_quantities = feature_tensor.shape
        assert df.shape == (n_data * n_z, n_quantities + 2)
        z = np.linspace(-np.pi, np.pi, n_z, endpoint=False)
        for j_data in range(n_data):
            np.testing.assert_allclose(
                df["j_tube"][j_data * n_z : (j_data + 1) * n_z],
                j_data,
            )
            np.testing.assert_allclose(df["z"][j_data * n_z : (j_data + 1) * n_z], z)
            for j_quantity in range(n_quantities):
                np.testing.assert_allclose(
                    df[names[j_quantity]][j_data * n_z : (j_data + 1) * n_z],
                    feature_tensor[j_data, :, j_quantity],
                )

        # Run tsfresh to make sure it isn't unhappy
        curated_tsfresh_features = {
            "maximum": None,
            "mean": None,
            "median": None,
            "minimum": None,
            "variance": None,
        }

        extracted_features = extract_features(
            df,
            column_id="j_tube",
            column_sort="z",
            default_fc_parameters=curated_tsfresh_features,
        )

    def test_drop_nearly_constant_features(self):
        features = make_test_dataframe()
        features2 = drop_nearly_constant_features(features)
        assert len(features2.columns) == len(features.columns) - 2

    def test_drop_special_characters_from_column_names(self):
        features = make_test_dataframe()
        features_array = features.to_numpy()
        features_columns = features.columns
        new_columns_should_be = [
            x.replace('"', "")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "_")
            .replace(" ", "_")
            for x in features.columns
        ]

        drop_special_characters_from_column_names(features)
        features2_array = features.to_numpy()
        features2_columns = features.columns

        np.testing.assert_allclose(features_array, features2_array)
        np.testing.assert_array_equal(new_columns_should_be, features2_columns)
        assert (
            features_columns[-1] != features2_columns[-1]
        )  # Last feature name should have been changed.

    def test_meaningful_names(self):
        _, names, _ = load_tensor("test")
        new_names = meaningful_names(names)
        np.testing.assert_equal(
            new_names,
            ['B', 'B⁻³𝗕×∇B⋅∇y', 'B⁻²𝗕×κ⋅∇y', 'B⁻³𝗕×∇B⋅∇x', '|∇y|²', '∇x⋅∇y', '|∇x|²'],
        )
