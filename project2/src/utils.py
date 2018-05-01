import math

from torch import FloatTensor, LongTensor

def build_data(n):
    """
    Builds a pair of tensors :
            - coordinates of shape (n, 2) (random coordinates in [0,1]x[0,1])
            - labels of shape (n, 1) (1 if random point is in disc of radius 1/sqrt(2*Pi), 0 else)
    :param n: number of points to sample
    :return: pair of tensors as described above.
    """
    # generate points uniformly at random in [0,1]^2
    coordinates = FloatTensor(n, 2).uniform_(0, 1)
    # create labels (shape (n,)
    labels = ((coordinates - FloatTensor([0.5, 0.5])).norm(p=2, dim=1) < 1 / math.sqrt(2 * math.pi)).type(LongTensor)
    # expand labels to one-hot encoding (shape (n,2)
    labels = FloatTensor(n, 2).zero_().scatter_(1, labels.view(-1, 1), 1)
    return coordinates, labels
