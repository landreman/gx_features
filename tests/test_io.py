import unittest
import numpy as np

from gx_features.io import load_all


class Tests(unittest.TestCase):

    def test_stride(self):

        data1 = load_all("20241005 small", verbose=False)
        data2 = load_all("20241005 small", verbose=False, stride=2)

        assert len(data2["Q"]) == (len(data1["Q"]) + 1) // 2
        np.testing.assert_allclose(data2["Q"], data1["Q"][::2])
        np.testing.assert_allclose(data2["Y"], data1["Y"][::2])
        np.testing.assert_allclose(data2["feature_tensor"], data1["feature_tensor"][::2, :, :])
        np.testing.assert_allclose(data2["scalar_feature_matrix"], data1["scalar_feature_matrix"][::2, :])
        