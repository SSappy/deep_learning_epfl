import numpy as np


def compute_accuracy(y, y_hat):
    return np.sum(y == y_hat)/np.size(y)


