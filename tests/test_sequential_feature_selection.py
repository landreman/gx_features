import unittest
import numpy as np

from gx_features.calculations import compute_reductions
from gx_features.io import load_tensor
from gx_features.sequential_feature_selection import (
    compute_features_20241107,
    reductions_20241108,
)


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

    def test_compute_reductions_2_ways(self):
        """reductions_20241108() should match compute_reductions()."""
        tensor, names, Y = load_tensor("test")
        n_data, n_z, n_quantities = tensor.shape

        quantiles = [0.4]
        count_above = [0.6]
        fft_coefficients = [3]
        n_reductions = reductions_20241108(
            1,
            1,
            quantiles=quantiles,
            count_above=count_above,
            fft_coefficients=fft_coefficients,
            return_n_reductions=True,
        )

        # names = simplify_names(names)
        extracted_features_1, _ = compute_reductions(
            tensor,
            names,
            max=True,
            min=True,
            max_minus_min=True,
            mean=True,
            median=True,
            rms=True,
            variance=True,
            skewness=True,
            quantiles=quantiles,
            count_above=count_above,
            fft_coefficients=fft_coefficients,
            mean_kpar=True,
            argmax_kpar=True,
            return_df=False,
        )
        np.testing.assert_equal(
            extracted_features_1.shape[1], n_reductions * n_quantities
        )
        for j_quantity in range(n_quantities):
            for j_reduction in range(n_reductions):
                # print(f"j_quantity: {j_quantity}  j_reduction: {j_reduction}")
                extracted_feature_2, _ = reductions_20241108(
                    tensor[:, :, j_quantity],
                    j_reduction,
                    quantiles=quantiles,
                    count_above=count_above,
                    fft_coefficients=fft_coefficients,
                )
                np.testing.assert_allclose(
                    extracted_features_1[:, j_reduction * n_quantities + j_quantity],
                    extracted_feature_2,
                )
