import unittest
import numpy as np

from gx_features.feature_sets import create_features_20240725_01


class Tests(unittest.TestCase):
    def test_create_features_20240725_01(self):
        features, names = create_features_20240725_01(test=True)
        assert len(names) == 180
        assert features.shape == (10, 96, 180)