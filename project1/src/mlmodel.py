from feature_augmentation import augment_features


class MLModel(object):
    """
    Base class for baseline and neural network models.
    """

    def update_data(self, data=None, targets=None, feature_augmentation=None):
        """
        Update the data set used to train the model.
        :param data: Raw data.
        :param targets: Labels of the data.
        :param feature_augmentation: Feature augmentation function applied to the data.
        :return: Nothing.
        """
        if targets is not None:
            self.targets = targets
            self.targets = self.targets.numpy()

        if feature_augmentation is not None:
            self.feature_augmentation = feature_augmentation

        if data is not None:
            self.data = data
            self.data = augment_features(self.data, self.feature_augmentation)

    def __init__(self):
        """
        Initializer.
        """
        self.targets = None
        self.feature_augmentation = None
        self.data = None
