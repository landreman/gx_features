import unittest
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import xgboost as xgb

from gx_features.calculations import compute_reductions
from gx_features.io import load_tensor, load_all
from gx_features.feature_sets import compute_fn_20241211
from gx_features.sequential_feature_selection import (
    compute_features_20241107,
    compute_fn_20241106,
    compute_fn_20241107,
    compute_fn_20241108,
    compute_fn_20241115,
    reductions_20241108,
    try_every_feature,
    sfs,
)


class DummyEstimator:
    """Dummy estimator for testing purposes."""

    _estimator_type = "regressor"

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
    def test_small_try_every_feature_mpi(self):
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
            np.testing.assert_allclose(
                results["best_score"], -0.11151405572891235, rtol=1e-6
            )

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

    def test_try_every_feature_fixed_features_mpi(self):
        data = load_all("20241005 small")
        Y = data["Y"]

        estimator = make_pipeline(StandardScaler(), xgb.XGBRegressor(n_jobs=1))
        from mpi4py import MPI

        # Try one fixed feature:
        index = 6
        fixed_feature = np.var(data["feature_tensor"][:, :, index], axis=1)
        results = try_every_feature(
            estimator,
            compute_fn_20241107,
            data,
            Y,
            fixed_features=fixed_feature.reshape((-1, 1)),
            verbose=1,
        )
        if MPI.COMM_WORLD.Get_rank() == 0:
            np.testing.assert_allclose(
                results["best_score"], 0.1272051692008972, rtol=1e-6
            )
            np.testing.assert_equal(results["best_feature_name"], "max(gds2)")

        # Try two fixed features:
        fixed_features = np.var(data["feature_tensor"][:, :, 5:6], axis=1)
        results = try_every_feature(
            estimator,
            compute_fn_20241107,
            data,
            Y,
            fixed_features=fixed_features,
            verbose=1,
        )
        if MPI.COMM_WORLD.Get_rank() == 0:
            np.testing.assert_allclose(
                results["best_score"], -0.19673858880996703, rtol=2e-6
            )
            np.testing.assert_equal(
                results["best_feature_name"], "max(gds22_over_shat_squared)"
            )

    def test_try_every_feature_20241108_mpi(self):
        data = load_all("20241005 small")
        Y = data["Y"]

        estimator = DummyEstimator(n_features=1)

        results = try_every_feature(estimator, compute_fn_20241108, data, Y, verbose=1)
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_rank() == 0:
            np.testing.assert_equal(len(results["names"]), 57270)

    def test_classifier_mpi(self):
        data = load_all("20241005 small")
        Q = data["Q"]
        Y = Q > np.mean(Q)

        estimator = xgb.XGBClassifier(n_features=1)

        results = try_every_feature(estimator, compute_fn_20241106, data, Y, verbose=1, scoring="neg_log_loss")
        # results = try_every_feature(estimator, compute_fn_20241106, data, Y, verbose=1, scoring="roc_auc")
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_rank() == 0:
            np.testing.assert_equal(len(results["names"]), 11)

    def test_parsimony(self):
        for parsimony_margin in [0, 1e-4]:
            data = load_all("20241005 small")
            Q = data["Q"]
            Y = Q > np.mean(Q)

            estimator = DummyEstimator(n_features=1)

            results = try_every_feature(estimator, compute_fn_20241106, data, Y, verbose=1, parsimony=True, parsimony_margin=parsimony_margin)
            from mpi4py import MPI

            if MPI.COMM_WORLD.Get_rank() == 0:
                np.testing.assert_equal(len(results["names"]), 11)
                np.testing.assert_equal(results["best_feature_name"], "nfp")
                np.testing.assert_equal(results["best_feature_index"], 7)

    def test_skip_single_features_mpi(self):
        data = load_all("20241005 small")
        # Use iota as the target feature, so iota would be selected if skip_single_feature=False
        index = data["scalars"].index("iota")
        Y = data["scalar_feature_matrix"][:, index]

        estimator = LinearRegression()

        results = try_every_feature(estimator, compute_fn_20241106, data, Y, verbose=1, skip_single_features=False)
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_rank() == 0:
            np.testing.assert_equal(len(results["names"]), 11)
            np.testing.assert_equal(results["best_feature_name"], "iota")
            np.testing.assert_equal(results["best_feature_index"], 8)

        # Now repeat with skip_single_features=True:
        results = try_every_feature(estimator, compute_fn_20241106, data, Y, verbose=1, skip_single_features=True)
        if MPI.COMM_WORLD.Get_rank() == 0:
            np.testing.assert_equal(len(results["names"]), 11)
            np.testing.assert_equal(results["best_feature_name"], "max(cvdrift)")
            np.testing.assert_equal(results["best_feature_index"], 2)

    def test_try_every_feature_Spearman_mpi(self):
        data = load_all("20241005 small")
        Y = data["Y"]

        results = try_every_feature("Spearman", compute_fn_20241108, data, Y, verbose=1)
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_rank() == 0:
            np.testing.assert_equal(len(results["names"]), 57270)

    def test_try_every_feature_20241211_mpi(self):
        data = load_all("20241005 small")
        Y = data["Y"]

        estimator = DummyEstimator(n_features=1)

        results = try_every_feature(estimator, compute_fn_20241211, data, Y, verbose=1)
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_rank() == 0:
            np.testing.assert_equal(len(results["names"]), 57274)

    @unittest.skip
    def test_try_every_feature_20241115_mpi(self):
        data = load_all("20241005 small")
        Y = data["Y"]

        estimator = DummyEstimator(n_features=1)

        results = try_every_feature(estimator, compute_fn_20241115, data, Y, verbose=1)

    def test_sfs_matches_try_every_feature_mpi(self):
        data = load_all("20241005 small")
        Y = data["Y"]

        estimator = make_pipeline(StandardScaler(), xgb.XGBRegressor(n_jobs=1))
        from mpi4py import MPI

        proc0 = MPI.COMM_WORLD.Get_rank() == 0

        n_steps = 3
        filename = "temp_sfs.pkl"
        sfs_results = sfs(estimator, compute_fn_20241107, data, Y, n_steps, verbose=1, filename=filename)

        fixed_features = None
        for j_step in range(n_steps):
            single_feature_results = try_every_feature(
                estimator,
                compute_fn_20241107,
                data,
                Y,
                fixed_features=fixed_features,
                verbose=1,
            )
            if proc0:
                for key in single_feature_results.keys():
                    print(
                        "Comparing",
                        key,
                        single_feature_results[key],
                        sfs_results[j_step][key],
                    )
                    np.testing.assert_equal(
                        single_feature_results[key], sfs_results[j_step][key]
                    )

                most_recent_best_feature = single_feature_results[
                    "best_feature"
                ].reshape((-1, 1))
                if j_step == 0:
                    fixed_features = most_recent_best_feature
                else:
                    fixed_features = np.hstack(
                        [fixed_features, most_recent_best_feature]
                    )
