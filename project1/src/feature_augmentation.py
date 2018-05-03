"""
File containing the feature augmentation functions used by the baseline models.
"""

from sklearn.preprocessing import PolynomialFeatures


def augment_polynomial(data, max_degree=5, interaction_only=False):
    """
    Augment a data matrix with polynomial basis.
    :param data: n*m real data matrix
    :param max_degree: Maximum degree of the polynomial base
    :param interaction_only: Product terms added or not.
    :return: Enhanced data with polynomial features
    """
    poly = PolynomialFeatures(max_degree, interaction_only=interaction_only)
    return poly.fit_transform(data)


def augment_features(data, feature_augmentation):
    """
    Augment features for a given data matrix.
    :param data: Data matrix.
    :param feature_augmentation: Function applied to augment the features.
    :return: Augmented data matrix.
    """
    if data is not None and feature_augmentation is not None:
        if isinstance(feature_augmentation, list):
            for augmentation_function in feature_augmentation:
                data = augmentation_function(data)
        else:
            data = feature_augmentation(data)
    return data
