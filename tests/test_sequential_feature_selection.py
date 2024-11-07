import unittest
import numpy as np

from gx_features.sequential_feature_selection import compute_features_20241107


class Tests(unittest.TestCase):
    def test_small_sfs_mpi(self):
        results = compute_features_20241107()

        if results is not None:
            # Results are only returned on rank 0

            assert results["names"] == [
                "min(bmag)",
                "max(bmag)",
                "min(gbdrift)",
                "max(gbdrift)",
                "min(cvdrift)",
                "max(cvdrift)",
                "min(gbdrift0_over_shat)",
                "max(gbdrift0_over_shat)",
                "min(gds2)",
                "max(gds2)",
                "min(gds21_over_shat)",
                "max(gds21_over_shat)",
                "min(gds22_over_shat_squared)",
                "max(gds22_over_shat_squared)",
                "nfp",
                "iota",
                "shat",
                "d_pressure_d_s",
            ]
            np.testing.assert_allclose(
                results["scores"],
                [
                    -1.24743416,
                    -0.7789896,
                    -1.43405595,
                    -0.86996129,
                    -0.78987402,
                    -0.8206272,
                    0.02390338,
                    -0.16311725,
                    -0.81661642,
                    -1.01757281,
                    -1.43039036,
                    -1.98132601,
                    -1.03986593,
                    -0.20190021,
                    -0.03772788,
                    -0.51621311,
                    -0.51270387,
                    -0.54295611,
                ],
                rtol=1e-5,
            )
            np.testing.assert_equal(results["best_feature_index"], 6)
            np.testing.assert_equal(
                results["best_feature_name"], "min(gbdrift0_over_shat)"
            )
            np.testing.assert_allclose(results["best_score"], 0.02390338182449341)
