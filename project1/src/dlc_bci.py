# This is distributed under BSD 3-Clause license

import torch
import numpy
import os
import errno

from six.moves import urllib


def tensor_from_file(root, filename,
                     base_url='https://documents.epfl.ch/users/f/fl/fleuret/www/data/bci'):
    """
    Loads a file with a tensor format and returns the corresponding torch.Tensor object. The files are downloaded
    if they are not in the root folder.
    :param root: (string) Name of the root directory
    :param filename: (string) Name of the file to read
    :param base_url: (string) Url from where the files are downloaded
    :return: A torch.Tensor object containing the file data
    """
    file_path = os.path.join(root, filename)

    if not os.path.exists(file_path):
        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        url = base_url + '/' + filename

        print('Downloading ' + url)
        data = urllib.request.urlopen(url)
        with open(file_path, 'wb') as f:
            f.write(data.read())

    return torch.from_numpy(numpy.loadtxt(file_path))


def load(root, train=True, download=True, one_khz=False):
    """
    Helper function to easily load the training or testing set with the right sampling frequency.
    :param root: (string) Name of the root directory
    :param train: (boolean) The training or the testing sets are loaded whether it is True or False respectively
    :param download: If True, downloads the dataset from the internet and puts it in root directory. If dataset is
                     already downloaded, it is not downloaded again.
    :param one_khz: The sampling frequency will be 100Hz or 1kHz if it is False or True respectively
    :return: The training or testing input and target
    """

    nb_electrodes = 28

    if train:

        if one_khz:
            dataset = tensor_from_file(root, 'sp1s_aa_train_1000Hz.txt')
        else:
            dataset = tensor_from_file(root, 'sp1s_aa_train.txt')

        input = dataset.narrow(1, 1, dataset.size(1) - 1)
        input = input.float().view(input.size(0), nb_electrodes, -1)
        target = dataset.narrow(1, 0, 1).clone().view(-1).long()

    else:

        if one_khz:
            input = tensor_from_file(root, 'sp1s_aa_test_1000Hz.txt')
        else:
            input = tensor_from_file(root, 'sp1s_aa_test.txt')
        target = tensor_from_file(root, 'labels_data_set_iv.txt')

        input = input.float().view(input.size(0), nb_electrodes, -1)
        target = target.view(-1).long()

    return input, target
