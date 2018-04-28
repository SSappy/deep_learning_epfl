from torch import FloatTensor
import numpy as np
import math

def build_data(n):
    """

    :param n:
    :return:
    """
    x = FloatTensor(np.random.rand(n, 2))
    y = np.linalg.norm((x - FloatTensor([0.5, 0.5])), axis=1) < 1 / math.sqrt(2 * math.pi)
    y = FloatTensor(y.astype(int).reshape((y.shape[0], 1)))
    return x, y