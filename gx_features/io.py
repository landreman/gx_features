import pickle
import os
import numpy as np


def load(test=False):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    test_file_dir = os.path.join(this_dir, "..", "tests", "files")
    if test:
        # File with the flux tube geometries (raw features):
        in_filename = os.path.join(test_file_dir, "test_data_in.pkl")
        # File with the GX heat flux
        out_filename = os.path.join(test_file_dir, "test_data_out.pkl")
    else:
        # File with the flux tube geometries (raw features):
        in_filename = (
            "20240601-01-assembleFluxTubeMatrix_noShiftScale_nz96_withCvdrift.pkl"
        )
        # File with the GX heat flux
        out_filename = "20240601-01-103_gx_nfp4_production_gx_results_gradxRemoved.pkl"

    with open(in_filename, "rb") as f:
        in_data = pickle.load(f)

    with open(out_filename, "rb") as f:
        out_data = pickle.load(f)

    raw_feature_matrix = in_data["matrix"]
    heat_flux_averages = out_data["Q_avgs_without_FSA_grad_x"]
    n_z = in_data["nl"]
    n_data = raw_feature_matrix.shape[0]
    n_quantities = in_data["n_quantities"]
    print("n_z:", n_z)
    print("n_features:", in_data["n_features"])
    print("n_quantities:", n_quantities)
    print("z_functions:", in_data["z_functions"])
    print("n_data:", n_data)

    Y = np.log(heat_flux_averages)
    print(in_data.keys())

    # Divide up the features so each quantity is a 3rd dimension in a tensor
    feature_tensor = np.zeros((n_data, n_z, n_quantities))
    for j in range(n_quantities):
        index = j * n_z
        feature_tensor[:, :, j] = raw_feature_matrix[:, index : index + n_z]

    data = {
        "Y": Y,
        "feature_tensor": feature_tensor,
        "n_z": n_z,
        "n_data": n_data,
        "n_quantities": n_quantities,
        "z_functions": in_data["z_functions"],
    }
    return data


def create_test_data(n=10):
    """Create a small dataset for testing, by taking the first n rows of the data."""

    # File with the flux tube geometries (raw features):
    in_filename = "/Users/mattland/Box/work24/20240601-01-assembleFluxTubeMatrix_noShiftScale_nz96_withCvdrift.pkl"
    # File with the GX heat flux
    out_filename = "/Users/mattland/Box/work24/20240601-01-103_gx_nfp4_production_gx_results_gradxRemoved.pkl"

    with open(in_filename, "rb") as f:
        in_data = pickle.load(f)

    with open(out_filename, "rb") as f:
        out_data = pickle.load(f)

    print("in_data.keys():", in_data.keys())
    for key, val in in_data.items():
        try:
            print(key, "shape:", val.shape)
        except:
            print(key, "scalar:")
    print("out_data.keys():", out_data.keys())
    for key, val in out_data.items():
        try:
            print(key, "shape:", val.shape)
        except:
            print(key, "scalar:")

    in_data["matrix"] = in_data["matrix"][:n, :]
    in_data["n_data"] = n
    in_data["tube_files"] = in_data["tube_files"][:n]

    out_data["tube_names"] = out_data["tube_names"][:n]
    out_data["VPrimes"] = out_data["VPrimes"][:n]
    out_data["FSA_grad_xs"] = out_data["FSA_grad_xs"][:n]
    out_data["Q_avgs_with_FSA_grad_x"] = out_data["Q_avgs_with_FSA_grad_x"][:n]
    out_data["Q_avgs_without_FSA_grad_x"] = out_data["Q_avgs_without_FSA_grad_x"][:n]

    this_dir = os.path.dirname(os.path.abspath(__file__))
    test_file_dir = os.path.join(this_dir, "..", "tests", "files")
    test_in_filename = os.path.join(test_file_dir, "test_data_in.pkl")
    test_out_filename = os.path.join(test_file_dir, "test_data_out.pkl")
    print("test_in_filename:", test_in_filename)
    with open(test_in_filename, "wb") as f:
        pickle.dump(in_data, f)

    with open(test_out_filename, "wb") as f:
        pickle.dump(out_data, f)
