import unittest
import numpy as np

from gx_features.feature_sets import (
    create_tensors_20240725_01,
    create_features_20240726_01,
    create_features_20240804_01,
    create_features_20240805_01,
    create_features_20240906_01,
    create_features_20241011_01,
    compute_fn_20241119,
    unary_funcs_20241123,
)
from gx_features.io import load_all
from gx_features.sequential_feature_selection import (
    try_every_feature,
    reductions_20241107,
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

    def test_create_features_20241011_01_mpi(self):
        df_mpi = create_features_20241011_01(10, mpi=True)
        df_no_mpi = create_features_20241011_01(10)

        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            assert list(df_no_mpi.columns) == list(df_mpi.columns)
            np.testing.assert_allclose(df_no_mpi.to_numpy(), df_mpi.to_numpy())

    @unittest.skip
    def test_compute_fn_20241119_mpi(self):
        data = load_all("20241005 small", verbose=False)
        Y = data["Y"]

        estimator = DummyEstimator(n_features=1)

        results = try_every_feature(estimator, compute_fn_20241119, data, Y, verbose=1)

    def test_compute_fn_20241119_mini_mpi(self):
        """Use a smaller set of unary functions and reductions to speed up the test."""

        data = load_all("20241005 small", verbose=False)
        Y = data["Y"]

        def compute_fn_20241119_mini(data, mpi_rank, mpi_size, evaluator):
            return compute_fn_20241119(
                data,
                mpi_rank,
                mpi_size,
                evaluator,
                unary_func=unary_funcs_20241123,
                reductions_func=reductions_20241107,
            )

        results = try_every_feature(
            "Spearman", compute_fn_20241119_mini, data, Y, verbose=1
        )

    def test_compute_fn_20241119_3_algorithms_mpi(self):
        """The three algorithms for compute_fn_20241119 should give identical answers."""

        def compute_fn_20241119_algorithm_1(data, mpi_rank, mpi_size, evaluator):
            return compute_fn_20241119(
                data,
                mpi_rank,
                mpi_size,
                evaluator,
                unary_func=unary_funcs_20241123,
                reductions_func=reductions_20241107,
                algorithm=1,
            )

        def compute_fn_20241119_algorithm_2(data, mpi_rank, mpi_size, evaluator):
            return compute_fn_20241119(
                data,
                mpi_rank,
                mpi_size,
                evaluator,
                unary_func=unary_funcs_20241123,
                reductions_func=reductions_20241107,
                algorithm=2,
            )

        def compute_fn_20241119_algorithm_3(data, mpi_rank, mpi_size, evaluator):
            return compute_fn_20241119(
                data,
                mpi_rank,
                mpi_size,
                evaluator,
                unary_func=unary_funcs_20241123,
                reductions_func=reductions_20241107,
                algorithm=3,
            )

        evaluator = "Spearman"

        data = load_all("20241005 small", verbose=False)
        Y = data["Y"]

        results1 = try_every_feature(
            evaluator, compute_fn_20241119_algorithm_1, data, Y, verbose=1
        )

        results2 = try_every_feature(
            evaluator, compute_fn_20241119_algorithm_2, data, Y, verbose=1
        )

        results3 = try_every_feature(
            evaluator, compute_fn_20241119_algorithm_3, data, Y, verbose=1
        )

        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_rank() != 0:
            return

        # The two algorithms return results in different orders, so we need to sort them before comparing.
        perm1 = np.argsort(results1["names"])
        perm2 = np.argsort(results2["names"])
        perm3 = np.argsort(results3["names"])
        np.testing.assert_equal(results1["names"][perm1], results2["names"][perm2])
        np.testing.assert_equal(results1["names"][perm1], results3["names"][perm3])
        np.testing.assert_allclose(
            results1["scores"][perm1], results2["scores"][perm2], rtol=1e-6
        )
        np.testing.assert_allclose(
            results1["scores"][perm1], results3["scores"][perm3], rtol=1e-6
        )

        for key in ["best_feature", "best_feature_name", "best_score"]:
            print(
                "Comparing",
                key,
                results1[key],
                results2[key],
                results3[key],
            )
            np.testing.assert_equal(results1[key], results2[key])
            np.testing.assert_equal(results1[key], results3[key])
