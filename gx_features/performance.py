import os
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
from sklearn.linear_model import Ridge, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


def assess_features_quick(features_filename, randomize_Y=False, include_kernel_ridge=False):
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
    estimator_xgb = make_pipeline(StandardScaler(), XGBRegressor())
    estimators = [
        estimator_knn,
        estimator_ridge,
        estimator_lgbm,
        estimator_xgb,
    ]
    estimator_names = ["10NN", "Ridge", "LightGBM", "XGBoost"]
    if include_kernel_ridge:
        estimators.append(estimator_kernel_ridge)
        estimator_names.append("Kernel Ridge")

    folds = KFold(n_splits=5, shuffle=True, random_state=0)
    # folds = KFold(n_splits=5, shuffle=False)

    original_R2s = []
    for estimator, name in zip(estimators, estimator_names):
        print(f"Cross-validation with {name}:")
        scores = cross_val_score(
            estimator, X_all, Y_all, cv=folds, scoring="r2", verbose=2
        )
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
        plt.plot(n_neighbors, scores_vs_param[j], ".-", label=f"outer fold {j}")

    plt.xlabel("n_neighbors")
    plt.ylabel("R^2")
    plt.title("K-nearest-neighbors - nested CV hyperparameter search")
    plt.legend(loc=0, fontsize=6)
    plt.figtext(
        0.5,
        0.005,
        os.path.abspath(features_filename),
        ha="center",
        va="bottom",
        fontsize=6,
    )
    plt.tight_layout()
    plt.show()


def hyperparam_search_ridge(features_filename):
    data = read_pickle(features_filename)
    Y_all = data["Y"]
    X_all = data.drop(columns="Y")
    # Y_all -= np.mean(Y_all)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_all, Y_all, test_size=0.2, random_state=0
    )

    alphas = 10.0 ** np.linspace(1, 3.2, 10)
    # alphas = [100, 300]
    param_name = "ridge__alpha"
    param_grid = {param_name: alphas}
    estimator = make_pipeline(StandardScaler(), Ridge())
    grid_search = GridSearchCV(estimator, param_grid, cv=5, verbose=2)
    grid_search.fit(X_train, Y_train)
    print("Best alpha:", grid_search.best_params_[param_name])
    print(f"Best R^2:        {grid_search.best_score_:.3}")
    print(f"R^2 on test set: {grid_search.score(X_test, Y_test):.3}")

    plt.semilogx(alphas, grid_search.cv_results_["mean_test_score"], ".-")
    plt.xlabel("alpha")
    plt.ylabel("R^2")
    plt.title("Ridge regression - hyperparameter search")
    plt.figtext(
        0.5,
        0.005,
        os.path.abspath(features_filename),
        ha="center",
        va="bottom",
        fontsize=6,
    )
    plt.tight_layout()
    plt.show()


def hyperparam_search_nested_ridge(features_filename):
    data = read_pickle(features_filename)
    Y_all = data["Y"].to_numpy()
    X_all = data.drop(columns="Y").to_numpy()
    # Y_all -= np.mean(Y_all)

    n_outer_splits = 5
    outer_folds = KFold(n_splits=n_outer_splits, shuffle=True, random_state=0)
    alphas = 10.0 ** np.linspace(1, 3.2, 10)
    # alphas = [100, 300]
    param_name = "ridge__alpha"
    param_grid = {param_name: alphas}
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

        estimator = make_pipeline(StandardScaler(), Ridge())
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
        plt.semilogx(alphas, scores_vs_param[j], ".-", label=f"outer fold {j}")

    plt.xlabel("alpha")
    plt.ylabel("R^2")
    plt.title("Ridge regression - nested CV hyperparameter search")
    plt.legend(loc=0, fontsize=6)
    plt.figtext(
        0.5,
        0.005,
        os.path.abspath(features_filename),
        ha="center",
        va="bottom",
        fontsize=6,
    )
    plt.tight_layout()
    plt.show()


def hyperparam_search_lasso(features_filename):
    data = read_pickle(features_filename)
    Y_all = data["Y"].to_numpy()
    X_all = data.drop(columns="Y").to_numpy()
    # Y_all -= np.mean(Y_all)

    # alphas = 10.0 ** np.linspace(-5, -3, 4)
    alphas = 10.0 ** np.linspace(-2, 0, 10)
    # alphas = [10]
    n_params = len(alphas)
    param_name = "lasso__alpha"
    param_grid = {param_name: alphas}
    n_folds = 5
    R2s = np.zeros((n_params, n_folds))
    n_features_used = np.zeros((n_params, n_folds))
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    for j_param, param in enumerate(alphas):
        print()
        print("#" * 80)
        print(f"alpha = {param}  (value {j_param + 1} / {n_params})")
        print("#" * 80)
        for j_fold, (train_index, test_index) in enumerate(folds.split(X_all)):
            print(f"\n --- Fold {j_fold + 1} / {n_folds} ---")
            print("Number of training samples:", len(train_index))
            print("Number of test samples:", len(test_index))
            X_train = X_all[train_index]
            Y_train = Y_all[train_index]
            X_test = X_all[test_index]
            Y_test = Y_all[test_index]

            estimator = make_pipeline(StandardScaler(), Lasso(alpha=param))
            estimator.fit(X_train, Y_train)
            R2 = estimator.score(X_test, Y_test)
            R2s[j_param, j_fold] = R2
            n_nonzero_coef = sum(np.abs(estimator.named_steps["lasso"].coef_) > 1e-13)
            print(f"R2: {R2}  n_nonzero_coef: {n_nonzero_coef}")
            n_features_used[j_param, j_fold] = n_nonzero_coef

    print("n_features_used:", n_features_used)
    avg_R2s = np.mean(R2s, axis=1)
    avg_n_features_used = np.mean(n_features_used, axis=1)

    print("Best alpha:", alphas[np.argmax(avg_R2s)])
    print(f"Best R^2:        {np.max(avg_R2s):.3}")

    plt.figure(figsize=(6, 8))
    nrows = 2
    ncols = 1

    plt.subplot(nrows, ncols, 1)
    plt.errorbar(alphas, avg_R2s, yerr=np.std(R2s, axis=1), fmt=".-")
    plt.xscale("log")
    plt.xlabel("alpha")
    plt.ylabel("R^2")
    plt.title("Lasso - hyperparameter search")
    
    plt.subplot(nrows, ncols, 2)
    plt.errorbar(alphas, avg_n_features_used, yerr=np.std(n_features_used, axis=1), fmt=".-")
    plt.xscale("log")
    plt.xlabel("alpha")
    plt.ylabel("# features kept")
    plt.ylim(bottom=0)
    
    plt.figtext(
        0.5,
        0.005,
        os.path.abspath(features_filename),
        ha="center",
        va="bottom",
        fontsize=6,
    )
    plt.tight_layout()
    plt.show()
