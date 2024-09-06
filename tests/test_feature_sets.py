import unittest
import numpy as np

from gx_features.feature_sets import (
    create_tensors_20240725_01,
    create_features_20240726_01,
    create_features_20240804_01,
    create_features_20240805_01,
    create_features_20240906_01,
)


class Tests(unittest.TestCase):
    def test_create_tensors_20240725_01(self):
        (
            single_quantity_tensor,
            single_quantity_names,
            combinations_tensor,
            combinations_names,
            Y,
        ) = create_tensors_20240725_01("test")
        assert len(single_quantity_names) == 8
        assert single_quantity_tensor.shape == (10, 96, 8)
        assert len(combinations_names) == 180
        assert combinations_tensor.shape == (10, 96, 180)
        assert Y.shape == (10,)

    def test_create_features_20240726_01(self):
        create_features_20240726_01("test")

    def test_create_features_20240804_01(self):
        create_features_20240804_01(10)

    def test_create_features_20240805_01(self):
        create_features_20240805_01(10)

    def test_create_features_20240906_01(self):
        create_features_20240906_01(10)
