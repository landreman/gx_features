import unittest
import numpy as np

from gx_features.calculations import compute_reductions
from gx_features.io import load_tensor, load_all
from gx_features.sequential_feature_selection import (
    compute_features_20241107,
    compute_fn_20241108,
    reductions_20241108,
    sfs,
)


class DummyEstimator:
    """Dummy estimator for testing purposes."""

    def __init__(self, n_features):
        self.n_features = n_features

    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.zeros(X.shape[0])

    def score(self, X, y):
        return 0.0

    def get_params(self, deep=False):
        return {"n_features": self.n_features}

    def set_params(self, **params):
        self.n_features = params["n_features"]
        return self


class Tests(unittest.TestCase):
    def test_small_sfs_mpi(self):
        results = compute_features_20241107()

        if results is not None:
            # Results are only returned on rank 0

            assert list(results["names"]) == [
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
                    -2.21912,
                    -1.623042,
                    -1.855183,
                    -1.522255,
                    -1.370325,
                    -1.327524,
                    -0.111514,
                    -0.277382,
                    -0.764641,
                    -1.548852,
                    -1.63394,
                    -2.044135,
                    -1.383299,
                    -0.534497,
                    -0.458563,
                    -1.981815,
                    -1.981366,
                    -1.981896,
                ],
                rtol=1e-5,
            )
            np.testing.assert_equal(results["best_feature_index"], 6)
            np.testing.assert_equal(
                results["best_feature_name"], "min(gbdrift0_over_shat)"
            )
            np.testing.assert_allclose(results["best_score"], -0.11151405572891235)

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

    def test_sfs_20241108_mpi(self):
        data = load_all("20241005 small")
        Y = data["Y"]

        estimator = DummyEstimator(n_features=1)

        results = sfs(estimator, compute_fn_20241108, data, Y, verbose=1)
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_rank() == 0:
            np.testing.assert_equal(len(results["names"]), 57270)
