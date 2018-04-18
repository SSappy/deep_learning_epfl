"""
File containing the feature augmentation functions used by the baseline models.
"""

from sklearn.preprocessing import PolynomialFeatures


def augment_polynomial(data, max_degree=5, interaction_only=False):
    """
    :param data: n*m real data matrix
    :param max_degree: Maximum degree of the polynomial base
    :param interaction_only:
    :return: Enhanced data with polynomial features
    """
    poly = PolynomialFeatures(max_degree, interaction_only=interaction_only)
    return poly.fit_transform(data)


def augment_data(data, feature_augmentation):
    """
    :param data:
    :param feature_augmentation:
    :return:
    """
    if data is not None and feature_augmentation is not None:
        if isinstance(feature_augmentation, list):
            for augmentation_function in feature_augmentation:
                data = augmentation_function(data)
        else:
            data = feature_augmentation(data)
    return data
