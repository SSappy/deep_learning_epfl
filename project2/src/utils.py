import math

from torch import FloatTensor
from numpy.linalg import norm
from numpy.random import rand
from numpy import ndarray

def build_data(n):
    """
    Builds a pair of tensors :
            - coordinates of shape (n, 2) (random coordinates in [0,1]x[0,1])
            - labels of shape (n, 1) (1 if random point is in disc of radius 1/sqrt(2*Pi), 0 else)
    :param n: number of points to sample
    :return: pair of tensors as described above.
    """
    coordinates = FloatTensor(rand(n, 2))
    test = norm((coordinates - FloatTensor([0.5, 0.5])), axis=1) < 1 / math.sqrt(2 * math.pi)
    labels = ndarray((n, 2))
    labels[test] = [0, 1]
    labels[~test] = [1, 0]
    labels = FloatTensor(labels)
    return coordinates, labels
