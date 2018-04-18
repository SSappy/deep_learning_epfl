from feature_augmentation_helper import augment_data


class MLModel():

    def update_data(self, data=None, targets=None, feature_augmentation=None):
        if targets is not None:
            self.targets = targets
            self.targets = self.targets.numpy()

        if feature_augmentation is not None:
            self.feature_augmentation = feature_augmentation

        if data is not None:
            self.data = data
            self.data = augment_data(self.data, self.feature_augmentation)

    def __init__(self):
        self.targets = None
        self.feature_augmentation = None
        self.data = None
