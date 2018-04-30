from torch.utils.data.dataset import Dataset


class BCIDataSet(Dataset):

    def __init__(self, data, targets, transform=None):
        self.transform = transform
        # Open and load text file including the whole training data
        self.__data = data
        self.__targets = targets

    # Override to give PyTorch access to any image on the data set
    def __getitem__(self, index):
        data = self.__data[index]
        if self.transform is not None:
            data = self.transform(data)
        target = self.__targets[index]
        return data, target

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__targets)
