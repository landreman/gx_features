import numpy as np
from pandas import DataFrame

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
            matrix_for_tsfresh[j * n_z : (j + 1) * n_z, j_quantity + 2] = tensor[j, :, j_quantity]

    columns = ["j_tube", "z"] + names
    df = DataFrame(matrix_for_tsfresh, columns=columns)
    return df
