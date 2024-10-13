import pickle
import os
import numpy as np

from .utils import in_github_actions


def load_all(dataset):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    test_file_dir = os.path.join(this_dir, "..", "tests", "files")
    data_dir = "/Users/mattland/Box/work24"
    # data_dir = "."
    if in_github_actions:
        data_dir = test_file_dir

    if dataset == "test":
        # File with the flux tube geometries (raw features):
        in_filename = os.path.join(test_file_dir, "test_data_in.pkl")
        # File with the GX heat flux
        out_filename = os.path.join(test_file_dir, "test_data_out.pkl")
    elif dataset == "20240601":
        # File with the flux tube geometries (raw features):
        in_filename = os.path.join(
            data_dir,
            "20240601-01-assembleFluxTubeMatrix_noShiftScale_nz96_withCvdrift.pkl",
        )
        # File with the GX heat flux
        out_filename = os.path.join(
            data_dir, "20240601-01-103_gx_nfp4_production_gx_results_gradxRemoved.pkl"
        )
    elif dataset == "20240726":
        # File with the flux tube geometries (raw features):
        in_filename = os.path.join(
            data_dir,
            "20240726-01-assembleFluxTubeTensor_vacuum_nz96.pkl",
        )
        # File with the GX heat flux
        out_filename = os.path.join(
            data_dir,
            "20240726-01-random_stellarator_equilibria_and_GX_gx_results_gradxRemoved.pkl",
        )
    elif dataset == "20241005":
        # File with the flux tube geometries (raw features):
        in_filename = os.path.join(
            data_dir,
            "20241004-01-assembleFluxTubeTensor_multiNfp_finiteBeta_nz96.pkl",
        )
        # File with the GX heat flux
        out_filename = os.path.join(
            data_dir,
            "20241004-01-random_stellarator_equilibria_GX_results_combined.pkl",
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    with open(in_filename, "rb") as f:
        in_data = pickle.load(f)

    with open(out_filename, "rb") as f:
        out_data = pickle.load(f)

    heat_flux_averages = out_data["Q_avgs_without_FSA_grad_x"]
    Y = np.log(heat_flux_averages)
    n_data = len(heat_flux_averages)

    print("in.data.keys():", in_data.keys())
    n_z = in_data["nl"]

    if dataset in ["test", "20240601"]:
        # These are old data that used "matrix" rather than "tensor"

        raw_feature_matrix = in_data["matrix"]

        # Divide up the features so each quantity is a 3rd dimension in a tensor
        n_quantities = in_data["n_quantities"]
        feature_tensor = np.zeros((n_data, n_z, n_quantities))
        for j in range(n_quantities):
            index = j * n_z
            feature_tensor[:, :, j] = raw_feature_matrix[:, index : index + n_z]
    else:
        feature_tensor = in_data["tensor"]

    print("n_z:", n_z)
    print("z_functions:", in_data["z_functions"])
    print("n_data:", n_data)
    assert len(Y) == feature_tensor.shape[0]

    data = {
        "Y": Y,
        "feature_tensor": feature_tensor,
        "n_z": n_z,
        "n_data": n_data,
    }
    fields_to_copy_from_input = [
        "z_functions", "scalars", "scalar_feature_matrix", "n_quantities",
    ]
    for f in fields_to_copy_from_input:
        if f in in_data.keys():
            data[f] = in_data[f]

    return data


def load_tensor(dataset):
    data = load_all(dataset)
    return data["feature_tensor"], data["z_functions"], data["Y"]


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


def create_test_data2():
    """Create small datasets for testing, by taking the first n rows of the data."""

    for dataset in ["20240601", "20240726", "20241005"]:
        print("\n\nProcessing dataset", dataset)

        data_dir = "/Users/mattland/Box/work24"
        if dataset == "20240601":
            # File with the flux tube geometries (raw features):
            in_filename = os.path.join(
                data_dir,
                "20240601-01-assembleFluxTubeMatrix_noShiftScale_nz96_withCvdrift.pkl",
            )
            # File with the GX heat flux
            out_filename = os.path.join(
                data_dir, "20240601-01-103_gx_nfp4_production_gx_results_gradxRemoved.pkl"
            )
            n = 20
        elif dataset == "20240726":
            # File with the flux tube geometries (raw features):
            in_filename = os.path.join(
                data_dir,
                "20240726-01-assembleFluxTubeTensor_vacuum_nz96.pkl",
            )
            # File with the GX heat flux
            out_filename = os.path.join(
                data_dir,
                "20240726-01-random_stellarator_equilibria_and_GX_gx_results_gradxRemoved.pkl",
            )
            n = 31
        elif dataset == "20241005":
            # File with the flux tube geometries (raw features):
            in_filename = os.path.join(
                data_dir,
                "20241004-01-assembleFluxTubeTensor_multiNfp_finiteBeta_nz96.pkl",
            )
            # File with the GX heat flux
            out_filename = os.path.join(
                data_dir,
                "20241004-01-random_stellarator_equilibria_GX_results_combined.pkl",
            )
            n = 43
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        with open(in_filename, "rb") as f:
            in_data = pickle.load(f)

        with open(out_filename, "rb") as f:
            out_data = pickle.load(f)

        old_n = len(in_data["tube_files"])
        print("in_data.keys():", in_data.keys())
        for key, val in in_data.items():
            if isinstance(val, list):
                print(key, "list:", len(val))
                if len(val) == old_n:
                    in_data[key] = val[:n]
                    print("  trimmed to:", len(in_data[key]))
            elif isinstance(val, np.ndarray):
                print(key, "shape:", val.shape)
                in_data[key] = val[:n]
                print("  trimmed to:", in_data[key].shape)
            else:
                print(key, "scalar:")
        print("out_data.keys():", out_data.keys())
        for key, val in out_data.items():
            if isinstance(val, list):
                print(key, "list:", len(val))
                if len(val) == old_n:
                    out_data[key] = val[:n]
                    print("  trimmed to:", len(out_data[key]))
            elif isinstance(val, np.ndarray):
                print(key, "shape:", val.shape)
                out_data[key] = val[:n]
                print("  trimmed to:", out_data[key].shape)
            else:
                print(key, "scalar:")

        # in_data["matrix"] = in_data["matrix"][:n, :]
        in_data["n_tubes"] = n
        # in_data["tube_files"] = in_data["tube_files"][:n]

        # out_data["tube_names"] = out_data["tube_names"][:n]
        # out_data["VPrimes"] = out_data["VPrimes"][:n]
        # out_data["FSA_grad_xs"] = out_data["FSA_grad_xs"][:n]
        # out_data["Q_avgs_with_FSA_grad_x"] = out_data["Q_avgs_with_FSA_grad_x"][:n]
        # out_data["Q_avgs_without_FSA_grad_x"] = out_data["Q_avgs_without_FSA_grad_x"][:n]

        this_dir = os.path.dirname(os.path.abspath(__file__))
        test_file_dir = os.path.join(this_dir, "..", "tests", "files")
        test_in_filename = os.path.join(test_file_dir, os.path.basename(in_filename)[:-4] + f"_rows{n}.pkl")
        test_out_filename = os.path.join(test_file_dir, os.path.basename(out_filename)[:-4] + f"_rows{n}.pkl")
        print("Saving trimmed input file:", test_in_filename)
        print("Saving trimmed output file:", test_out_filename)
        with open(test_in_filename, "wb") as f:
            pickle.dump(in_data, f)

        with open(test_out_filename, "wb") as f:
            pickle.dump(out_data, f)
