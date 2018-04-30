import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F

from mlmodel import MLModel

from bci_dataset import BCIDataSet
from data_augmentation_helper import augment_data

from sklearn.preprocessing import MinMaxScaler

from torchvision import transforms

from preprocessing import Normalize
from data_augmentation_helper import GaussianNoise
from data_augmentation_helper import Crop1d
from data_augmentation_helper import Resize


class NNModel(MLModel, nn.Module):

    def update_data(self, data=None, targets=None, feature_augmentation=None):
        MLModel.update_data(self, data, targets, feature_augmentation)

        if data is not None:
            self.data = self.data.view(self.data.shape[0], 1, self.data.shape[1])

            self.normalizer = Normalize(torch.min(self.data), torch.max(self.data))
            self.train_transform = transforms.Compose([GaussianNoise(0, 1),
                                                       # Crop1d(),
                                                       # Resize(self.data.shape[2]),
                                                       self.normalizer])
            self.test_transform = transforms.Compose([self.normalizer])
            self.data_set = BCIDataSet(self.data, targets, transform=self.train_transform)

    def __init__(self, data=None, targets=None, feature_augmentation=None):
        MLModel.__init__(self)
        nn.Module.__init__(self)
        self.data_var = None
        self.targets_var = None
        self.normalizer = None
        self.train_transform = None
        self.test_transform = None
        self.data_set = None
        self.update_data(data=data, targets=targets, feature_augmentation=feature_augmentation)

    def forward(self, x):
        return

    def fit(self, data=None, targets=None, epochs=30, batch_size=128, optimizer='adam', lr=0.01, momentum=0):
        self.train()

        if isinstance(data, torch.utils.data.dataset.Dataset):
            data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                                      shuffle=True, num_workers=2)
        else:
            self.update_data(data=data, targets=targets)
            data_loader = torch.utils.data.DataLoader(self.data_set, batch_size=batch_size,
                                                      shuffle=True, num_workers=2)

        optimizer = optimizer.lower()
        if optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        else:
            raise ValueError('The argument "optimizer" is invalid')

        criterion = nn.CrossEntropyLoss()

        epoch = 0

        try:
            for epoch in range(epochs):  # loop over the data set multiple times
                running_loss = 0.0
                for i, data in enumerate(data_loader):
                    # get the inputs
                    inputs, labels = data
                    if i == 0 or i == 20:
                        pass
                        # print(inputs.shape)
                        # print('min {0} max {1}'.format(np.min(inputs), np.max(inputs)))
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    if i % 2000 == 1999:  # print every 2000 mini-batches
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / 2000))
                        running_loss = 0.0
        except KeyboardInterrupt:
            print('Interrupted at epoch {}.'.format(epoch))

    def predict(self, data, raw=False):
        self.eval()
        data = data.view(data.shape[0], 1, -1)
        data = self.test_transform(data)
        outputs = self(data)
        if not raw:
            _, outputs = torch.max(outputs.data, dim=1)
        return outputs

    def compute_accuracy(self, x_test, y_test):
        y_hat = self.predict(x_test).numpy()
        y_test = y_test.numpy()
        return np.sum(y_test == y_hat)/np.size(y_test)