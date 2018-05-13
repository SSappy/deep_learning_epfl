"""
This file contains several classes and functions related to data augmentation  :
- class GaussianNoise
- class Crop1d
- function downsample
"""

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
        if not self.size:
            return data

        begin = randint(0, data.shape[-1] - self.size)
        if len(data.shape) == 2:
            return data[:, begin:(begin + self.size)]
        else:
            return data[:, :, begin:(begin + self.size)]


def downsample(data, targets, size=50, regular=True, count=10):
    length = data.shape[2]
    step = length//size
    downsampled_data = torch.Tensor()
    downsampled_targets = torch.LongTensor()
    if regular:
        for start in range(step):
            indexes = range(start, length, step)
            downsampled_data = torch.cat((downsampled_data, data[:, :, indexes]), 0)
            downsampled_targets = torch.cat((downsampled_targets, targets), 0)

            if start >= count - 1:
                break
    else:
        for _ in range(count):
            ks = [randint(step) for _ in range(size)]
            indexes = [k + i for k, i in zip(ks, list(range(0, length, step)))]
            downsampled_data = torch.cat((downsampled_data, data[:, :, indexes]), 0)
            downsampled_targets = torch.cat((downsampled_targets, targets), 0)

    return downsampled_data, downsampled_targets
