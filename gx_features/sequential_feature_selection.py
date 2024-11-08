# This module contains functions for a custom version of forward sequential
# feature selection which is parallelized using MPI and in which it is not necessary to store all the features.


import numpy as np
from scipy.stats import skew
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from .calculations import compute_mean_k_parallel
from .io import load_all


def reductions_20241107(arr, j):
    if j == 0:
        return arr.min(axis=1), "min"
    elif j == 1:
        return arr.max(axis=1), "max"
    else:
        raise ValueError(f"Invalid reduction index: {j}")


def reductions_20241108(
    arr,
    j,
    quantiles=[0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
    count_above=np.arange(-2, 6.1, 0.5),
    fft_coefficients=[1, 2, 3],
    return_n_reductions=False,
):
    n_quantiles = len(quantiles)
    n_count_above = len(count_above)
    n_fft_coefficients = len(fft_coefficients)
    n_reductions = 10 + n_quantiles + n_count_above + n_fft_coefficients
    if return_n_reductions:
        return n_reductions

    if j == 0:
        return arr.max(axis=1), "max"

    elif j == 1:
        return arr.min(axis=1), "min"

    elif j == 2:
        return arr.max(axis=1) - arr.min(axis=1), "maxMinusMin"

    elif j == 3:
        return arr.mean(axis=1), "mean"

    elif j == 4:
        return np.median(arr, axis=1), "median"

    elif j == 5:
        return np.sqrt(np.mean(arr**2, axis=1)), "rootMeanSquare"

    elif j == 6:
        return np.var(arr, axis=1), "variance"

    elif j == 7:
        return np.nan_to_num(skew(arr, axis=1, bias=False)), "skewness"

    elif j in range(8, 8 + n_quantiles):
        q = quantiles[j - 8]
        return np.quantile(arr, q, axis=1), f"quantile{q}"

    elif j in range(8 + n_quantiles, 8 + n_quantiles + n_count_above):
        i = j - 8 - n_quantiles
        print(f"j: {j}  i: {i}  count_above[i]: {count_above[i]}")
        return np.mean(arr > count_above[i], axis=1), f"countAbove{count_above[i]}"

    elif j in range(
        8 + n_quantiles + n_count_above,
        8 + n_quantiles + n_count_above + n_fft_coefficients,
    ):
        abs_fft_result = np.abs(np.fft.fft(arr, axis=1))
        i = j - 8 - n_quantiles - n_count_above
        print(f"j: {j}  i: {i}  fft_coefficients[i]: {fft_coefficients[i]}")
        return (
            abs_fft_result[:, fft_coefficients[i]],
            f"absFFTCoeff{fft_coefficients[i]}",
        )

    elif j == 8 + n_quantiles + n_count_above + n_fft_coefficients:
        new_features, kpar_names = compute_mean_k_parallel(
            arr.reshape((arr.shape[0], arr.shape[1], 1)), [""], include_argmax=False
        )
        return new_features[:, 0], "meanKpar"

    elif j == 9 + n_quantiles + n_count_above + n_fft_coefficients:
        new_features, kpar_names = compute_mean_k_parallel(
            arr.reshape((arr.shape[0], arr.shape[1], 1)), [""], include_argmax=True
        )
        return new_features[:, 1], "argmaxKpar"

    else:
        raise ValueError(f"Invalid reduction index: {j}")


def compute_fn_20241107(data, mpi_rank, mpi_size, evaluator):
    z_functions = data["z_functions"]
    feature_tensor = data["feature_tensor"]
    scalars = data["scalars"]
    scalar_feature_matrix = data["scalar_feature_matrix"]
    n_z_functions = len(z_functions)

    index = 0

    # Apply all possible reductions to all of the original z-functions:
    for j_z_function in range(n_z_functions):
        z_function_data = feature_tensor[:, :, j_z_function]
        z_function_name = z_functions[j_z_function]
        for j_reduction in range(2):
            if index % mpi_size == mpi_rank:
                reduction, reduction_name = reductions_20241107(
                    z_function_data, j_reduction
                )
                evaluator(reduction, f"{reduction_name}({z_function_name})", index)
            index += 1

    # Try all the extra scalar features:
    n_scalars = len(scalars)
    for j in range(n_scalars):
        evaluator(scalar_feature_matrix[:, j], scalars[j], index)
        index += 1


def compute_fn_20241108(data, mpi_rank, mpi_size, evaluator):
    z_functions = data["z_functions"]
    feature_tensor = data["feature_tensor"]
    scalars = data["scalars"]
    scalar_feature_matrix = data["scalar_feature_matrix"]
    n_z_functions = len(z_functions)

    reductions_func = reductions_20241108

    oneOverB = 1 / feature_tensor[:, :, 0]
    n_reductions = reductions_func(1, 1, return_n_reductions=True)

    index = 0

    for include_extra_oneOverB in [False, True]:
        # Apply all possible reductions to all of the original z-functions:
        for j_z_function in range(n_z_functions):
            z_function_data = feature_tensor[:, :, j_z_function]
            z_function_name = z_functions[j_z_function]
            if include_extra_oneOverB:
                z_function_data *= oneOverB
                z_function_name += "_/_B"

            for j_reduction in range(n_reductions):
                if index % mpi_size == mpi_rank:
                    reduction, reduction_name = reductions_func(
                        z_function_data, j_reduction
                    )
                    evaluator(reduction, f"{reduction_name}({z_function_name})", index)
                index += 1

    # Try all the extra scalar features:
    n_scalars = len(scalars)
    for j in range(n_scalars):
        evaluator(scalar_feature_matrix[:, j], scalars[j], index)
        index += 1


def compute_features_20241107():
    print("About to load data", flush=True)
    data = load_all("20241005 small")
    print("Done loading load data", flush=True)
    Y = data["Y"]

    estimator = make_pipeline(StandardScaler(), xgb.XGBRegressor(n_jobs=1))

    results = sfs(estimator, compute_fn_20241107, data, Y)
    return results


def sfs(estimator, compute_fn, data, Y):
    # To do later: >1 feature, fixed_features, possibly backtracking?

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()
    proc0 = mpi_rank == 0

    local_names = []
    local_scores = []
    local_indices = []
    local_best_feature = None
    local_best_feature_name = None
    local_best_feature_index = -1
    local_best_score = -1

    def evaluator(feature, name, index):
        # If this mpi process owns this feature, then compute the R^2 score and
        # store results.
        nonlocal local_best_feature, local_best_feature_name, local_best_feature_index, local_best_score, local_indices

        if index % mpi_size != mpi_rank:
            return

        score_arr = cross_val_score(estimator, feature.reshape((-1, 1)), Y, cv=5)
        score = score_arr.mean()
        local_names.append(name)
        local_scores.append(score)
        local_indices.append(index)
        print(f"[{mpi_rank}] index {index}: {name}, score: {score}", flush=True)
        if local_best_feature is None or score > local_best_score:
            local_best_feature = feature
            local_best_feature_name = name
            local_best_feature_index = index
            local_best_score = score

    # Each MPI rank does its share of the calculations:
    compute_fn(data, mpi_rank, mpi_size, evaluator)

    # Send results to proc 0:
    if not proc0:
        comm.send(local_names, dest=0)
        comm.send(local_scores, dest=0)
        comm.send(local_indices, dest=0)
        comm.send(local_best_feature, dest=0)
        comm.send(local_best_feature_name, dest=0)
        comm.send(local_best_feature_index, dest=0)
        comm.send(local_best_score, dest=0)
        return

    # Remaining tasks are done only on proc 0:
    names = local_names
    scores = local_scores
    indices = local_indices
    best_features = [local_best_feature]
    best_feature_names = [local_best_feature_name]
    best_feature_indices = [local_best_feature_index]
    best_scores = [local_best_score]
    for rank in range(1, mpi_size):
        names += comm.recv(source=rank)
        scores += comm.recv(source=rank)
        indices += comm.recv(source=rank)
        best_features.append(comm.recv(source=rank))
        best_feature_names.append(comm.recv(source=rank))
        best_feature_indices.append(comm.recv(source=rank))
        best_scores.append(comm.recv(source=rank))

    permutation = np.argsort(indices)
    indices = np.array(indices)[permutation]
    scores = np.array(scores)[permutation]
    names = list(np.array(names)[permutation])

    # Find the best feature across all ranks:
    best_rank = np.argmax(best_scores)
    best_feature = best_features[best_rank]
    best_feature_name = best_feature_names[best_rank]
    best_feature_index = best_feature_indices[best_rank]
    best_score = best_scores[best_rank]

    print("names:", names)
    print("scores:", scores)
    for name, score in zip(names, scores):
        print(f"{score:6.3f}  {name}")

    print(
        f"best_scores: {best_scores}  best_rank: {best_rank}  best_score: {best_score}"
    )
    print(f"best_feature_names: {best_feature_names}")
    print(f"best_feature_name: {best_feature_name}")
    print(f"best_feature_indices: {best_feature_indices}")
    print(f"best_feature_index: {best_feature_index}")

    results = {
        "names": names,
        "scores": scores,
        "best_feature": best_feature,
        "best_feature_name": best_feature_name,
        "best_feature_index": best_feature_index,
        "best_score": best_score,
        "best_features": best_features,
        "best_feature_names": best_feature_names,
        "best_feature_indices": best_feature_indices,
        "best_scores": best_scores,
    }
    return results
