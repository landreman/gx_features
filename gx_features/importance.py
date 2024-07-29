import numpy as np
from pandas import read_pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
import lightgbm as lgb


def plot_lgbm_importance(features_filename):
    data = read_pickle(features_filename)
    importance_filename = features_filename[:-4] + "_importances.npy"
    Y_all = data["Y"].to_numpy()
    X_all = data.drop(columns="Y")
    feature_names = X_all.columns
    X_all = X_all.to_numpy()
    n_features = X_all.shape[1]

    # estimator = make_pipeline(StandardScaler(), lgb.LGBMRegressor())
    # estimator.fit(X_all, Y_all)
    # lgb.plot_importance(
    #     estimator.named_steps["lgbmregressor"],
    #     # importance_type="gain",
    #     max_num_features=40,
    #     figsize=(14, 7.5),
    # )
    n_folds = 2
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    importances = np.zeros((n_folds, n_features))
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
        importances[j_fold, :] = estimator.named_steps["lgbmregressor"].feature_importances_

    avg_importances = np.mean(importances, axis=0)
    print("Saving importances as", importance_filename)
    np.save(importance_filename, avg_importances)

    importance_threshold = 1
    print(f"\nFeatures with importance < {importance_threshold}:")
    for j in range(n_features):
        if avg_importances[j] < importance_threshold:
            print(feature_names[j])

    order = np.argsort(avg_importances)[::-1]

    n_features_to_plot = 30
    figsize = (14.5, 8.5)

    plt.figure(figsize=figsize)
    ticks = range(n_features_to_plot)
    plt.barh(ticks, avg_importances[order[:n_features_to_plot]])
    plt.yticks(ticks, [feature_names[order[j]] for j in order[:n_features_to_plot]])
    plt.title("Most important features")
    plt.xlabel("Importance from LightGBM")
    plt.tight_layout()

    plt.figure(figsize=figsize)
    ticks = range(n_features)
    plt.barh(ticks, avg_importances[order])
    # plt.barh(ticks, avg_importances[order[-n_features_to_plot:]])
    # plt.yticks(ticks, [feature_names[order[-n_features_to_plot + j]] for j in order[:n_features_to_plot]])
    plt.title("All features")
    plt.xlabel("Importance from LightGBM")
    plt.ylabel("Feature index")
    plt.tight_layout()

    plt.show()
