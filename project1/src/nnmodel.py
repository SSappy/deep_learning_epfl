import torch.nn as nn

from mlmodel import MLModel

from feature_augmentation_helper import augment_data


class NNModel(MLModel, nn.Module):

    def update_data(self, data=None, targets=None, feature_augmentation=None):
        MLModel.update_data(self, data, targets, feature_augmentation)

    def __init__(self, data=None, targets=None, feature_augmentation=None):
        MLModel.__init__(self)
        super(NNModel, self).__init__()
        self.update_data(data=data, targets=targets, feature_augmentation=feature_augmentation)

    def forward(self, x):
        return x

    def train(self):
        raise NotImplementedError

    def predict(self, data):
        data = augment_data(data, self.feature_augmentation)

        raise NotImplementedError
