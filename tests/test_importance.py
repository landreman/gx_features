import unittest
import os
import numpy as np

from gx_features.importance import plot_SFS_results

this_dir = os.path.dirname(os.path.abspath(__file__))
test_file_dir = os.path.join(this_dir, "files")


class Tests(unittest.TestCase):
    def test_plot_SFS_results(self):
        filename = os.path.join(
            test_file_dir,
            "20240801-01-028_forward_sequential_feature_selection_SFS.pkl",
        )
        plot_SFS_results(filename, show=False)
