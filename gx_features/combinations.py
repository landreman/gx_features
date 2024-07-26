import numpy as np

from .calculations import differentiate


def remove_cvdrift(feature_tensor):
    """Remove the cvdrift quantity from the feature tensor."""
    # Shape of feature tensor: (n_data, n_z, n_quantities)
    assert feature_tensor.ndim == 3
    assert feature_tensor.shape[2] == 7

    new_feature_tensor = np.zeros((feature_tensor.shape[0], feature_tensor.shape[1], 6))
    new_feature_tensor[:, :, :2] = feature_tensor[:, :, :2]
    new_feature_tensor[:, :, 2:] = feature_tensor[:, :, 3:]

    return new_feature_tensor


def add_local_shear(feature_tensor, names, include_integral=True):
    """Adds the local shear and (optionally) the integrated local shear to the
    data.

    If you are going to call remove_cvdrift(), it must be done before calling this function.
    """
    # Shape of feature tensor: (n_data, n_z, n_quantities)
    assert feature_tensor.ndim == 3
    assert feature_tensor.shape[2] == 7 or feature_tensor.shape[2] == 6

    n_data, n_z, n_quantities = feature_tensor.shape

    n_quantities_new = n_quantities + 1
    names += ["localShear"]
    if include_integral:
        n_quantities_new += 1
        names += ["integratedLocalShear"]

    new_feature_tensor = np.zeros((n_data, n_z, n_quantities_new))
    new_feature_tensor[:, :, :n_quantities] = feature_tensor
    # z_functions: ['bmag', 'gbdrift', 'cvdrift', 'gbdrift0_over_shat', 'gds2', 'gds21_over_shat', 'gds22_over_shat_squared']
    integrated_local_shear = feature_tensor[:, :, -2] / feature_tensor[:, :, -1]

    new_feature_tensor[:, :, n_quantities] = differentiate(integrated_local_shear)
    if include_integral:
        new_feature_tensor[:, :, -1] = integrated_local_shear

    return new_feature_tensor, names


def create_masks(feature_tensor):
    n_data, n_z, n_quantities = feature_tensor.shape

    if n_quantities == 6:
        # cvdrift was removed
        masks = np.ones((n_data, n_z, 3))
        masks[:, :, 1] = feature_tensor[:, :, 1] >= 0
        masks[:, :, 2] = feature_tensor[:, :, 1] <= 0
        mask_names = ["", "gbdriftPos", "gbdriftNeg"]

    elif n_quantities == 7:
        # cvdrift was not removed
        masks = np.ones((n_data, n_z, 5))
        masks[:, :, 1] = feature_tensor[:, :, 1] >= 0
        masks[:, :, 2] = feature_tensor[:, :, 1] <= 0
        masks[:, :, 3] = feature_tensor[:, :, 2] >= 0
        masks[:, :, 4] = feature_tensor[:, :, 2] <= 0
        mask_names = ["", "gbdriftPos", "gbdriftNeg", "cvdriftPos", "cvdriftNeg"]
    else:
        raise ValueError(
            f"Wrong number of quantities in feature tensor: {n_quantities}."
        )

    return masks, mask_names


def make_feature_mask_combinations(feature_tensor, quantity_names, masks, mask_names):
    n_data, n_z, n_quantities = feature_tensor.shape
    assert masks.shape[0] == n_data
    assert masks.shape[1] == n_z
    n_masks = masks.shape[2]
    mask_names_with_underscores = []
    for mask_name in mask_names:
        if mask_name == "":
            mask_names_with_underscores.append("")
        else:
            mask_names_with_underscores.append("_" + mask_name)

    feature_mask_combinations = np.zeros((n_data, n_z, n_quantities * n_masks))
    names = []
    for i in range(n_masks):
        feature_mask_combinations[:, :, i * n_quantities : (i + 1) * n_quantities] = (
            feature_tensor * masks[:, :, i][:, :, None]
        )
        for quantity_name in quantity_names:
            names.append(quantity_name + mask_names_with_underscores[i])

    return feature_mask_combinations, names


def make_feature_product_combinations(feature_tensor, names):
    n_data, n_z, n_quantities = feature_tensor.shape
    n_combinations = (n_quantities * (n_quantities - 1)) // 2
    feature_product_combinations = np.zeros((n_data, n_z, n_combinations))
    combination_names = []
    j = 0
    for i in range(n_quantities):
        for k in range(i + 1, n_quantities):
            feature_product_combinations[:, :, j] = (
                feature_tensor[:, :, i] * feature_tensor[:, :, k]
            )
            combination_names.append(names[i] + "_" + names[k])
            j += 1
    assert j == n_combinations

    return feature_product_combinations, combination_names


# def combine_features(tensor1, names1, tensor2, names2):
#     n_data, n_z, n_quantities1 = tensor1.shape
#     n_data2, n_z2, n_quantities2 = tensor2.shape
#     assert n_data == n_data2
#     assert n_z == n_z2

#     combined_tensor = np.concatenate((tensor1, tensor2), axis=2)
#     combined_names = names1 + names2

#     return combined_tensor, combined_names


def combine_tensors(*args):
    assert (
        len(args) % 2 == 0
    ), "combine_features must have an even number of arguments: tensor1, names1, tensor2, names2, etc"
    tensors = args[::2]
    names = args[1::2]
    n_data, n_z, n_quantities1 = tensors[0].shape
    for tensor in tensors:
        n_data2, n_z2, n_quantities2 = tensor.shape
        assert n_data == n_data2
        assert n_z == n_z2

    combined_tensor = np.concatenate(tensors, axis=2)
    combined_names = sum(names, start=[])

    return combined_tensor, combined_names
