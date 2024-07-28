import numpy as np
from pandas import DataFrame
from tsfresh import extract_features
from feature_engine.selection import DropConstantFeatures

from .io import load_tensor


def tensor_to_tsfresh_dataframe(tensor, names):
    """
    Convert a numpy rank-3 tensor with quantities vs z into a DataFrame with
    the format expected by tsfresh.
    """
    n_data, n_z, n_quantities = tensor.shape
    matrix_for_tsfresh = np.zeros((n_data * n_z, n_quantities + 2))
    z = np.linspace(-np.pi, np.pi, n_z, endpoint=False)
    for j in range(n_data):
        matrix_for_tsfresh[j * n_z : (j + 1) * n_z, 0] = j
        matrix_for_tsfresh[j * n_z : (j + 1) * n_z, 1] = z
        for j_quantity in range(n_quantities):
            matrix_for_tsfresh[j * n_z : (j + 1) * n_z, j_quantity + 2] = tensor[
                j, :, j_quantity
            ]

    columns = ["j_tube", "z"] + names
    df = DataFrame(matrix_for_tsfresh, columns=columns)
    return df


def make_test_dataframe():
    feature_tensor, names, Y = load_tensor(True)
    n_quantities = 2
    feature_tensor = feature_tensor[:, :, :n_quantities]
    names = names[:n_quantities]
    df = tensor_to_tsfresh_dataframe(feature_tensor, names)
    curated_tsfresh_features = {
        "maximum": None,
        "mean": None,
        "minimum": None,
        "count_above": [{"t": 100.0}],
    }

    extracted_features = extract_features(
        df,
        column_id="j_tube",
        column_sort="z",
        default_fc_parameters=curated_tsfresh_features,
    )
    return extracted_features


def drop_nearly_constant_features(X, tol=0.95):
    dcf = DropConstantFeatures(tol=tol)
    X_nearly_constant_features_removed = dcf.fit_transform(X)
    print(
        f"Dropping features that have the same value for a fraction {tol} of the data."
    )
    print("  Number of features dropped:", len(dcf.features_to_drop_))
    print("  Features that were dropped:")
    for f in dcf.features_to_drop_:
        print(f)
    return X_nearly_constant_features_removed
