import re
import numpy as np
from pandas import DataFrame, read_pickle
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
    feature_tensor, names, Y = load_tensor("test")
    n_quantities = 2
    feature_tensor = feature_tensor[:, :, :n_quantities]
    names = names[:n_quantities]
    df = tensor_to_tsfresh_dataframe(feature_tensor, names)
    curated_tsfresh_features = {
        "maximum": None,
        "mean": None,
        "minimum": None,
        "count_above": [{"t": 100.0}],
        "fft_coefficient": [{"attr": "abs", "coeff": 1}],
    }

    from tsfresh import extract_features
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


def drop_special_characters_from_column_names(X):
    """
    LightGBM complains about some of the column names generated by tsfresh.
    To address this issue, this routine removes the special characters from the column names.
    """

    def renamer(x):
        return (
            x.replace('"', "")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "_")
            .replace(" ", "_")
        )

    X.rename(renamer, axis="columns", inplace=True)
    return X


def row_subset(filename, n):
    """
    Given a feature file (in pandas .pkl format), pick the first n rows and save
    the result as another pandas pkl file.
    """
    df = read_pickle(filename)
    df_subset = df.iloc[:n]
    new_filename = filename[:-4] + f"_rows{n}.pkl"
    df_subset.to_pickle(new_filename)
    print(f"Saved {n} rows to {new_filename}")


def simplify_names(names):
    """
    Given a list of names, return a simplified version of the names.
    """
    simplified_names = []
    for name in names:
        simplified_name = name.replace("_over_shat_squared", "").replace(
            "_over_shat", ""
        )
        simplified_names.append(simplified_name)
    return simplified_names

def meaningful_names(names):
    """
    Given a list of GX variable names, return the names in regular physics notation.
    """
    new_names = []
    for n in names:
        # Note that gbdrift0 must come before gbdrift, and gds21 & gds22 must
        # come before gds2.
        n = n.replace("bmag", "B")
        n = n.replace("gbdrift0_over_shat", "B⁻³𝗕×∇B⋅∇x")
        n = n.replace("gbdrift", "B⁻³𝗕×∇B⋅∇y")
        n = n.replace("cvdrift", "B⁻²𝗕×κ⋅∇y")
        n = n.replace("gds21_over_shat", "∇x⋅∇y")
        n = n.replace("gds22_over_shat_squared", "|∇x|²")
        n = n.replace("gds2", "|∇y|²")
        n = n.replace("localShear", "S")
        new_names.append(n)

    return new_names
