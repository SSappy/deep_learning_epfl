import torch

from numpy.random import randint


class GaussianNoise(object):
    """
    Data augmentation class to add a gaussian noise to the data.
    """

    def __init__(self, mean=0, std=1):
        """
        Initializer.
        :param mean: Mean of the added gaussian noise.
        :param std: Standard deviation of the added gaussian noise.
        """
        self.mean = mean
        self.std = std

    def __call__(self, data):
        """
        Add noises to a given data point.
        :param data: Data to transform.
        :return: Noised data.
        """
        noise = torch.FloatTensor(data.shape).normal_(self.mean, self.std)
        return data + noise


class Crop1d(object):
    """
    Data augmentation class to crop a time series randomly with a given window size.
    """

    def __init__(self, size=42):
        """
        Initializer.
        :param size: Size of the window.
        """
        self.size = size

    def __call__(self, data):
        """
        Crops a given data point.
        :param data: Data to transform.
        :return: Cropped data.
        """
        begin = randint(0, data.shape[1] - self.size)
        return data[:, begin:(begin + self.size)]
