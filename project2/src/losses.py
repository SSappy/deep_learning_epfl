from numpy import mean
from torch import FloatTensor

class MSE:
    def __init__(self):
        pass

    @staticmethod
    def compute_loss(pred, target):
        assert pred.shape == target.shape
        pred = FloatTensor(pred)
        return mean(((target-pred)**2).numpy())

    @staticmethod
    def compute_grad(pred, target):
        assert pred.shape == target.shape
        return 2/pred.shape[0] * (pred - target)