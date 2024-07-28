import numpy as np
import matplotlib.pyplot as plt
from pandas import read_pickle
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    GridSearchCV,
)
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor


def assess_features_quick(features_filename, randomize_Y=False):
    """
    randomize_Y is a sanity check: if we randomly permute the heat fluxes, there
    should be no true signal for the ML model to learn, so the R^2 should be close to zero.
    """
    data = read_pickle(features_filename)
    Y_all = data["Y"]
    X_all = data.drop(columns="Y")
    Y_all -= np.mean(Y_all)
    if randomize_Y:
        Y_all = np.random.permutation(Y_all)

    best_ridge_alpha = 268.26957952797216

    best_kernel_ridge_alpha = 0.028840315031266057
    best_kernel_ridge_gamma = 0.00021134890398366476

    estimator_ridge = make_pipeline(StandardScaler(), Ridge(alpha=best_ridge_alpha))
    estimator_kernel_ridge = make_pipeline(
        StandardScaler(),
        KernelRidge(
            kernel="rbf", alpha=best_kernel_ridge_alpha, gamma=best_kernel_ridge_gamma
        ),
    )
    estimator_knn = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10))
    estimator_lgbm = make_pipeline(StandardScaler(), LGBMRegressor())
    estimators = [
        estimator_ridge,
        estimator_kernel_ridge,
        estimator_knn,
        estimator_lgbm,
    ]
    estimator_names = ["Ridge", "Kernel Ridge", "10NN", "LightGBM"]
    folds = KFold(n_splits=5, shuffle=True, random_state=0)
    # folds = KFold(n_splits=5, shuffle=False)

    original_R2s = []
    for estimator, name in zip(estimators, estimator_names):
        print(f"Cross-validation with {name}:")
        scores = cross_val_score(estimator, X_all, Y_all, cv=folds, scoring="r2")
        R2 = scores.mean()
        print(f"    scores:", scores)
        print(f"    R^2: {R2:.3}")
        original_R2s.append(R2)

    print()
    for j in range(len(estimators)):
        print(f"    R^2 with {estimator_names[j]}: {original_R2s[j]:.3}")


def hyperparam_search_knn(features_filename):
    data = read_pickle(features_filename)
    Y_all = data["Y"]
    X_all = data.drop(columns="Y")
    # Y_all -= np.mean(Y_all)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_all, Y_all, test_size=0.2, random_state=0
    )

    n_neighbors = np.arange(1, 21)
    param_name = "kneighborsregressor__n_neighbors"
    param_grid = {param_name: n_neighbors}
    estimator = make_pipeline(StandardScaler(), KNeighborsRegressor())
    grid_search = GridSearchCV(estimator, param_grid, cv=5, verbose=2)
    grid_search.fit(X_train, Y_train)
    print("Best n_neighbors:", grid_search.best_params_[param_name])
    print(f"Best R^2:        {grid_search.best_score_:.3}")
    print(f"R^2 on test set: {grid_search.score(X_test, Y_test):.3}")

    plt.plot(n_neighbors, grid_search.cv_results_["mean_test_score"], ".-")
    plt.xlabel("n_neighbors")
    plt.ylabel("R^2")
    plt.title("KNN hyperparameter search")
    plt.tight_layout()
    plt.show()


def hyperparam_search_nested_knn(features_filename, max_n_neighbors=20):
    data = read_pickle(features_filename)
    Y_all = data["Y"].to_numpy()
    X_all = data.drop(columns="Y").to_numpy()
    # Y_all -= np.mean(Y_all)

    n_outer_splits = 5
    outer_folds = KFold(n_splits=n_outer_splits, shuffle=True, random_state=0)
    n_neighbors = np.arange(1, max_n_neighbors + 1)
    # n_neighbors = [1, 5, 10, 15]
    param_name = "kneighborsregressor__n_neighbors"
    param_grid = {param_name: n_neighbors}
    scores_vs_param = []
    R2s = []
    best_params = []
    for j_outer, (train_index, test_index) in enumerate(outer_folds.split(X_all)):
        print("#" * 80)
        print(f"Outer fold {j_outer + 1} / {n_outer_splits}")
        print("#" * 80)
        print("Number of training samples:", len(train_index))
        print("Number of test samples:", len(test_index))
        X_train = X_all[train_index]
        Y_train = Y_all[train_index]
        X_test = X_all[test_index]
        Y_test = Y_all[test_index]
        assert len(Y_train) == X_train.shape[0]
        assert len(Y_test) == X_test.shape[0]
        assert len(Y_train) + len(Y_test) == len(Y_all)
        assert len(Y_train) == len(train_index)
        assert len(Y_test) == len(test_index)

        estimator = make_pipeline(StandardScaler(), KNeighborsRegressor())
        grid_search = GridSearchCV(estimator, param_grid, cv=5, verbose=2)
        grid_search.fit(X_train, Y_train)

        R2 = grid_search.score(X_test, Y_test)
        best_param = grid_search.best_params_[param_name]

        R2s.append(R2)
        scores_vs_param.append(grid_search.cv_results_["mean_test_score"])
        best_params.append(best_param)

        print("Best n_neighbors:", best_param)
        print(f"Best R^2:        {grid_search.best_score_:.3}")
        print(f"R^2 on test set: {R2:.3}")

    print("#" * 80)
    print("Best parameters for each outer fold:", best_params)
    print("R^2 for each outer fold:", R2s)
    print("Mean R^2:", np.mean(R2s))

    for j in range(n_outer_splits):
        plt.plot(n_neighbors, scores_vs_param[j], ".-")

    plt.xlabel("n_neighbors")
    plt.ylabel("R^2")
    plt.title("K-nearest-neighbors hyperparameter search")
    plt.tight_layout()
    plt.show()
