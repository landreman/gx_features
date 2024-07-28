import pickle
import numpy as np
from pandas import read_pickle
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold
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
    X_all = data.drop(columns="Y")
    Y_all = data["Y"]
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
    estimators = [estimator_ridge, estimator_kernel_ridge, estimator_knn, estimator_lgbm]
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

# def hyperparam_search_knn(features_file):
