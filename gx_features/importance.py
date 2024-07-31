import pickle
import numpy as np
from pandas import read_pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso
import lightgbm as lgb
import xgboost as xgb


def plot_importances(features_filename, ridge_alpha=20, lasso_alpha=1e-3):
    data = read_pickle(features_filename)
    importance_filename = features_filename[:-4] + "_importances.pkl"
    Y_all = data["Y"].to_numpy()
    X_all = data.drop(columns="Y")
    feature_names = X_all.columns
    X_all = X_all.to_numpy()
    n_features = X_all.shape[1]

    importance_types = ["ridge", "lasso", "lightGBM_gain", "XGBoost_gain"]
    n_importance_types = len(importance_types)

    n_folds = 5
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    importances = np.zeros((n_folds, n_features, n_importance_types))
    R2s = np.zeros((n_folds, n_importance_types))
    for j_fold, (train_index, test_index) in enumerate(folds.split(X_all)):
        print("#" * 80)
        print(f"Fold {j_fold + 1} / {n_folds}")
        print("#" * 80)
        print("Number of training samples:", len(train_index))
        print("Number of test samples:", len(test_index))
        X_train = X_all[train_index]
        Y_train = Y_all[train_index]
        X_test = X_all[test_index]
        Y_test = Y_all[test_index]

        estimator = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
        estimator.fit(X_train, Y_train)
        R2 = estimator.score(X_test, Y_test)
        print("Ridge R2:", R2)
        j_importance_type = 0
        assert importance_types[j_importance_type] == "ridge"
        R2s[j_fold, j_importance_type] = R2
        importances[j_fold, :, j_importance_type] = np.abs(
            estimator.named_steps["ridge"].coef_
        )

        estimator = make_pipeline(StandardScaler(), Lasso(alpha=lasso_alpha))
        estimator.fit(X_train, Y_train)
        R2 = estimator.score(X_test, Y_test)
        print("Lasso R2:", R2)
        j_importance_type = 1
        assert importance_types[j_importance_type] == "lasso"
        R2s[j_fold, j_importance_type] = R2
        importances[j_fold, :, j_importance_type] = np.abs(
            estimator.named_steps["lasso"].coef_
        )

        estimator = make_pipeline(StandardScaler(), lgb.LGBMRegressor())
        estimator.fit(X_train, Y_train)
        R2 = estimator.score(X_test, Y_test)
        print("LightGBM R2:", R2)
        j_importance_type = 2
        assert importance_types[j_importance_type] == "lightGBM_gain"
        R2s[j_fold, j_importance_type] = R2
        importances[j_fold, :, j_importance_type] = estimator.named_steps[
            "lgbmregressor"
        ].booster_.feature_importance(importance_type="gain")

        estimator = make_pipeline(
            StandardScaler(), xgb.XGBRegressor(importance_type="gain")
        )
        estimator.fit(X_train, Y_train)
        R2 = estimator.score(X_test, Y_test)
        print("XGBoost R2:", R2)
        j_importance_type = 3
        assert importance_types[j_importance_type] == "XGBoost_gain"
        R2s[j_fold, j_importance_type] = R2
        importances[j_fold, :, j_importance_type] = estimator.named_steps[
            "xgbregressor"
        ].feature_importances_

    print("R2s:", R2s)
    R2s = np.mean(R2s, axis=0)
    print("Mean R2:", R2s)

    avg_importances = np.mean(importances, axis=0)
    print("Saving importances as", importance_filename)
    data = {
        "importance_types": importance_types,
        "n_importance_types": n_importance_types,
        "importances": avg_importances,
        "ridge_alpha": ridge_alpha,
        "lasso_alpha": lasso_alpha,
        "feature_names": feature_names,
        "n_features": n_features,
        "R2s": R2s,
    }
    with open(importance_filename, "wb") as f:
        pickle.dump(data, f)

    # importance_threshold = 0.1
    # for j_importance_type, importance_type in enumerate(importance_types):
    #     print(f"\nFeatures with {importance_type} importance < {importance_threshold}:")
    #     for j in range(n_features):
    #         if avg_importances[j, j_importance_type] < importance_threshold:
    #             print(feature_names[j])

    n_features_to_plot = 40
    figsize = (14.5, 8.5)

    for j_importance_type, importance_type in enumerate(importance_types):
        print("Importance type:", importance_type)

        order = np.argsort(avg_importances[:, j_importance_type])[::-1]
        print("order[0]:", order[0])
        print("feature_names[order[0]]:", feature_names[order[0]])
        print(
            "avg_importances[order[0], j_importance_type]:",
            avg_importances[order[0], j_importance_type],
        )

        plt.figure(figsize=figsize)
        ticks = range(n_features_to_plot)
        plt.barh(ticks, avg_importances[order[:n_features_to_plot], j_importance_type])
        plt.yticks(ticks, [feature_names[j] for j in order[:n_features_to_plot]])
        plt.title("Most important features")
        plt.xlabel(f"{importance_type} importance")
        plt.tight_layout()

        plt.figure(figsize=figsize)
        ticks = range(n_features)
        plt.barh(ticks, avg_importances[order, j_importance_type])
        plt.title("All features")
        plt.xlabel(f"{importance_type} importance")
        plt.ylabel("Feature index")
        plt.tight_layout()

    plt.show()

def plot_feature_selection_1st_cut(features_filename):
    data = read_pickle(features_filename)
    # importance_filename = features_filename[:-4] + "_importances.pkl"
    Y_all = data["Y"].to_numpy()
    X_all = data.drop(columns="Y")
    feature_names = X_all.columns
    X_all = X_all.to_numpy()
    n_features = X_all.shape[1]

    n_features_to_try = [int(round(n)) for n in np.linspace(1, n_features, 10)]
    n_features_to_try += [int(round(n)) for n in np.linspace(1, 60, 5)]
    n_features_to_try += [int(round(n)) for n in np.linspace(1, 40, 10)]
    n_features_to_try += [int(round(n)) for n in np.arange(1, 20)]
    n_features_to_try = sorted(list(set(n_features_to_try)))
    print("n_features_to_try:", n_features_to_try)
    n_n_features_to_try = len(n_features_to_try)

    # Find the importances, averaged over all CV folds:
    n_folds = 5
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    importances = np.zeros((n_folds, n_features))
    R2s = np.zeros(n_folds)
    for j_fold, (train_index, test_index) in enumerate(folds.split(X_all)):
        print("#" * 80)
        print(f"Fold {j_fold + 1} / {n_folds}")
        print("#" * 80)
        print("Number of training samples:", len(train_index))
        print("Number of test samples:", len(test_index))
        X_train = X_all[train_index]
        Y_train = Y_all[train_index]
        X_test = X_all[test_index]
        Y_test = Y_all[test_index]

        estimator = make_pipeline(StandardScaler(), lgb.LGBMRegressor())
        estimator.fit(X_train, Y_train)
        R2 = estimator.score(X_test, Y_test)
        print("LightGBM R2:", R2)
        R2s[j_fold] = R2
        importances[j_fold, :] = estimator.named_steps[
            "lgbmregressor"
        ].booster_.feature_importance(importance_type="gain")

    avg_importances = importances.mean(axis=0)
    order = np.argsort(avg_importances)[::-1]

    R2s = np.zeros(n_n_features_to_try)
    stds = np.zeros(n_n_features_to_try)
    for j, n_features_reduced in enumerate(n_features_to_try):
        print("#" * 80)
        print(f"Trying {n_features_reduced} features")
        print("#" * 80)
        # print("Features to keep:")
        # for j in range(n_features_reduced):
        #     print(feature_names[order[j]])
        X_reduced = X_all[:, order[:n_features_reduced]]

        n_folds = 5
        folds = KFold(n_splits=n_folds, shuffle=True, random_state=0)
        R2s_fold = np.zeros(n_folds)
        for j_fold, (train_index, test_index) in enumerate(folds.split(X_reduced)):
            print("#" * 80)
            print(f"Fold {j_fold + 1} / {n_folds}")
            print("#" * 80)
            print("Number of training samples:", len(train_index))
            print("Number of test samples:", len(test_index))
            X_train = X_reduced[train_index]
            Y_train = Y_all[train_index]
            X_test = X_reduced[test_index]
            Y_test = Y_all[test_index]

            estimator = make_pipeline(StandardScaler(), lgb.LGBMRegressor())
            estimator.fit(X_train, Y_train)
            R2 = estimator.score(X_test, Y_test)
            print("LightGBM R2:", R2)
            R2s_fold[j_fold] = R2

        R2s[j] = R2s_fold.mean()
        stds[j] = R2s_fold.std()

    plt.figure()
    plt.errorbar(n_features_to_try, R2s, yerr=stds, fmt='.-')
    plt.xlabel("Number of features")
    plt.ylabel("R2")
    plt.tight_layout()
    plt.show()