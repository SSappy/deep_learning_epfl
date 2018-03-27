from mlmodel import MLModel


class NNModel(MLModel):

    def __init__(self):
        MLModel.__init__(self)
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError
