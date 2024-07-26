from .io import load
from .combinations import add_local_shear, create_masks, make_feature_mask_combinations, make_feature_product_combinations, combine_tensors

def create_features_20240725_01(test=False):
    data = load(test)
    feature_tensor = data["feature_tensor"]
    names = data["z_functions"]

    # Create masks:
    masks, mask_names = create_masks(feature_tensor)

    # Add local shear as a feature:
    feature_tensor, names = add_local_shear(feature_tensor, names, include_integral=False)

    # Create feature-pair products:
    feature_product_tensor, feature_product_names = make_feature_product_combinations(
        feature_tensor, names
    )

    # Create single-feature-mask combinations:
    feature_mask_tensor, feature_mask_names = make_feature_mask_combinations(
        feature_tensor, names, masks, mask_names
    )
    
    # Create feature-pair-mask combinations:
    feature_pair_mask_tensor, feature_pair_mask_names = make_feature_mask_combinations(
        feature_product_tensor, feature_product_names, masks, mask_names
    )
    
    final_tensor, final_names = combine_tensors(
        feature_mask_tensor,
        feature_mask_names,
        feature_pair_mask_tensor,
        feature_pair_mask_names,
    )

    print("Number of features:", len(final_names))
    for n in final_names:
        print(n)

    return final_tensor, final_names
