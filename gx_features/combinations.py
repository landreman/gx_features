import numpy as np

from .calculations import differentiate


def remove_cvdrift(feature_tensor, names):
    """Remove the cvdrift quantity from the feature tensor."""
    # Shape of feature tensor: (n_data, n_z, n_quantities)
    assert feature_tensor.ndim == 3
    assert feature_tensor.shape[2] == 7

    new_feature_tensor = np.zeros((feature_tensor.shape[0], feature_tensor.shape[1], 6))
    new_feature_tensor[:, :, :2] = feature_tensor[:, :, :2]
    new_feature_tensor[:, :, 2:] = feature_tensor[:, :, 3:]

    new_names = names[:2] + names[3:]

    return new_feature_tensor, new_names


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
    # Make a copy of the "names" list so that we don't modify the original list.
    names = names.copy() + ["localShear"]
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


def create_threshold_masks(
    feature_matrix, name="gbdrift", thresholds=np.arange(-1, 1.05, 0.05)
):
    n_data, n_z = feature_matrix.shape
    n_masks = len(thresholds)

    masks = np.ones((n_data, n_z, n_masks))
    mask_names = []
    for i, threshold in enumerate(thresholds):
        masks[:, :, i] = feature_matrix >= threshold
        mask_names.append(f"{name}GtrThan{threshold:.2f}")

    return masks, mask_names


def make_pairwise_products_from_2_sets(
    feature_tensor1, names1, feature_tensor2, names2
):
    n_data, n_z, n_quantities1 = feature_tensor1.shape
    assert feature_tensor2.shape[0] == n_data
    assert feature_tensor2.shape[1] == n_z
    n_quantities2 = feature_tensor2.shape[2]
    names2_with_underscores = []
    for name in names2:
        if name == "":
            names2_with_underscores.append("")
        else:
            names2_with_underscores.append("_" + name)

    combinations_tensor = np.zeros((n_data, n_z, n_quantities1 * n_quantities2))
    names = []
    for i in range(n_quantities2):
        combinations_tensor[:, :, i * n_quantities1 : (i + 1) * n_quantities1] = (
            feature_tensor1 * feature_tensor2[:, :, i][:, :, None]
        )
        for quantity_name in names1:
            names.append(quantity_name + names2_with_underscores[i])

    return combinations_tensor, names


def make_feature_product_combinations(feature_tensor, names):
    n_data, n_z, n_quantities = feature_tensor.shape
    n_combinations = (n_quantities * (n_quantities - 1)) // 2
    tensor_product_combinations = np.zeros((n_data, n_z, n_combinations))
    combination_names = []
    j = 0
    for i in range(n_quantities):
        for k in range(i + 1, n_quantities):
            if names[i] == "1/" + names[k] or names[k] == "1/" + names[i]:
                # Don't multiply a feature by its own inverse
                n_combinations -= 1
                continue

            tensor_product_combinations[:, :, j] = (
                feature_tensor[:, :, i] * feature_tensor[:, :, k]
            )
            combination_names.append(names[i] + "_x_" + names[k])
            j += 1
    assert j == n_combinations

    tensor_product_combinations = tensor_product_combinations[:, :, :n_combinations]

    return tensor_product_combinations, combination_names


def make_feature_quotient_combinations(feature_tensor, names):
    n_data, n_z, n_quantities = feature_tensor.shape
    n_positive_quantities = 0
    positive_quantity_indices = []
    for j in range(n_quantities):
        if np.all(feature_tensor[:, :, j] > 0):
            n_positive_quantities += 1
            positive_quantity_indices.append(j)

    # Number of quotient combinations:
    # Any of n_positive_quantities can be the denominator.
    # The numerator can be any of the quantities except the denominator.
    n_combinations = n_positive_quantities * (n_quantities - 1)
    tensor_quotient_combinations = np.zeros((n_data, n_z, n_combinations))
    combination_names = []
    new_index = 0
    for j_denominator_positive in range(n_positive_quantities):
        j_denominator_global = positive_quantity_indices[j_denominator_positive]

        for j_numerator in range(n_quantities):
            if j_numerator == j_denominator_global:
                continue

            tensor_quotient_combinations[:, :, new_index] = (
                feature_tensor[:, :, j_numerator]
                / feature_tensor[:, :, j_denominator_global]
            )
            combination_names.append(
                names[j_numerator] + "_/_" + names[j_denominator_global]
            )
            new_index += 1

    assert new_index == n_combinations

    return tensor_quotient_combinations, combination_names


def make_inverse_quantities(feature_tensor, names):
    n_data, n_z, n_quantities = feature_tensor.shape
    positive_quantity_indices = []
    new_names = []
    for j in range(n_quantities):
        if np.all(feature_tensor[:, :, j] > 0):
            positive_quantity_indices.append(j)
            new_names.append("1/" + names[j])

    new_tensor = 1 / feature_tensor[:, :, positive_quantity_indices]
    return new_tensor, new_names


def make_feature_product_and_quotient_combinations(feature_tensor, names):
    tensor_product_combinations, product_names = make_feature_product_combinations(
        feature_tensor, names
    )
    tensor_quotient_combinations, quotient_names = make_feature_quotient_combinations(
        feature_tensor, names
    )
    return combine_tensors(
        tensor_product_combinations,
        product_names,
        tensor_quotient_combinations,
        quotient_names,
    )


def divide_by_quantity(feature_tensor, names, quotient, quotient_name):
    """
    Divide each quantity in the feature tensor by the given quantity.

    feature_tensor: (n_data, n_z, n_quantities)
    quotient: (n_data, n_z)
    """

    indices_to_include = []
    new_names = []
    for j, name in enumerate(names):
        if not name.startswith(quotient_name):
            indices_to_include.append(j)
            new_names.append(names[j] + "_/_" + quotient_name)

    new_features = feature_tensor[:, :, indices_to_include] / quotient[:, :, None]
    return new_features, new_names


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


def heaviside_transformations(tensor, names, long_names=False):
    """
    For every quantity x in the tensor that takes on both positive and negative
    values, create new quantities H(x) and H(-x), where H is the Heaviside step
    function.
    """
    n_data, n_z, n_quantities = tensor.shape

    quantities_with_both_signs = []
    new_names = []
    for j in range(n_quantities):
        if np.any(tensor[:, :, j] > 0) and np.any(tensor[:, :, j] < 0):
            quantities_with_both_signs.append(j)
            if long_names:
                temp_name = names[j].replace("B⁻²", "").replace("B⁻³", "")
                new_names.append("Heaviside(" + temp_name + ")")
                new_names.append("Heaviside(-" + temp_name + ")")
            else:
                new_names.append(names[j] + "Pos")
                new_names.append(names[j] + "Neg")

    n_quantities_with_both_signs = len(quantities_with_both_signs)

    new_tensor = np.zeros((n_data, n_z, 2 * n_quantities_with_both_signs))
    for j in range(n_quantities_with_both_signs):
        new_tensor[:, :, 2 * j] = np.heaviside(
            tensor[:, :, quantities_with_both_signs[j]], 0
        )
        new_tensor[:, :, 2 * j + 1] = np.heaviside(
            -tensor[:, :, quantities_with_both_signs[j]], 0
        )

    return new_tensor, new_names
