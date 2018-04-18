from mlmodel import MLModel

from feature_augmentation_helper import augment_data


class NNModel(MLModel):

    def update_data(self, data=None, targets=None, feature_augmentation=None):
        MLModel.update_data(self, data, targets, feature_augmentation)

    def __init__(self, data=None, targets=None, feature_augmentation=None):
        MLModel.__init__(self)
        self.update_data(data=data, targets=targets, feature_augmentation=feature_augmentation)

    def train(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError
