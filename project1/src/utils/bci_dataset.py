"""
This file defines the class BCIDataSet inheriting torch.utils.data.dataset.DataSet. It is a container
the data set used for the training in nnmodel.py.
"""

from torch.utils.data.dataset import Dataset


class BCIDataSet(Dataset):
    """
    Class inheriting the Dataset class from torch.utils.data.dataset. It is used to make a data loader from the data
    and train the models easily.
    """

    def __init__(self, data, targets, transform=None):
        """
        Initializer of the class.
        :param data: Raw data set.
        :param targets: Labels of the data.
        :param transform: Transformations applied to the data (e.g. data augmentation).
        """
        self.transform = transform
        # Open and load text file including the whole training data
        self.__data = data
        self.__targets = targets

    # Override to give PyTorch access to any image on the data set
    def __getitem__(self, index):
        """
        Method returning a data point given an index after applying the transformations on it.
        :param index: Index of the data point.
        :return: Data point at this index.
        """
        data = self.__data[index]
        if self.transform is not None:
            data = self.transform(data)
        target = self.__targets[index]
        return data, target

    # Override to give PyTorch size of data set
    def __len__(self):
        """
        Method used to know the length of the data set.
        :return: The length of the data set.
        """
        return len(self.__targets)
