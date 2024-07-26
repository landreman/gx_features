import unittest
import numpy as np
from tsfresh import extract_features

from gx_features.io import load_tensor
from gx_features.utils import tensor_to_tsfresh_dataframe


class Tests(unittest.TestCase):
    def test_tensor_to_tsfresh_dataframe(self): 
        feature_tensor, names = load_tensor(True)
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
            np.testing.assert_allclose(
                df["z"][j_data * n_z : (j_data + 1) * n_z],
                z
            )
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