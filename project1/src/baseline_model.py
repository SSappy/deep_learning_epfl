from mlmodel import MLModel


class BaselineModel(MLModel):

    def __init__(self):
        MLModel.__init__(self)
        raise NotImplementedError

    def train(self, feature_augmentation=None):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError
