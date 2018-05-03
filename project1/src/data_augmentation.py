import numpy as np

import torch
from torch import Tensor

import librosa

from numpy.random import randint
from scipy.signal import resample


class GaussianNoise(object):
    """
    Add a gaussian noise to the data.
    """

    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        noise = torch.FloatTensor(data.shape).normal_(self.mean, self.std)
        return data + noise


class Crop1d(object):
    """
    Add a gaussian noise to the data.
    """

    def __init__(self, size=42):
        self.size = size

    def __call__(self, data):
        begin = randint(0, data.shape[1] - self.size)
        return data[:, begin:(begin + self.size)]


class Resize(object):
    """
    Add a gaussian noise to the data.
    """

    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, data):
        res = Tensor(resample(data.numpy(), self.new_size))
        print(res.shape)
        return res


def augment_data(data, method):
    data_transform = torch.FloatTensor(data.shape[0], data.shape[1], data.shape[2])
    if method == 'noise':
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                # wn = Tensor(np.random.randn(len(data[i, j, :])))
                # data_transform[i, j, :] = data[i, j, :] + 10*wn
                data_transform[i, j, :] = data[i, j, :] + torch.FloatTensor(np.random.normal(0, 1, data[i, j, :].shape))
    if method == 'roll':
        data_transform = torch.FloatTensor(np.roll(data, 10, axis=2))
    if method == 'stretch':
        rate = 0.6
        input_length = 50
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data1 = librosa.effects.time_stretch(np.array(data[i, j, :]), rate)
                if len(data1) > input_length:
                    data1 = data1[:input_length]
                else:
                    data1 = np.pad(data1, (0, max(0, input_length - len(data1))), 'constant')
                data_transform[i, j, :] = torch.FloatTensor(data1)
    return torch.cat([data, data_transform], 0)
