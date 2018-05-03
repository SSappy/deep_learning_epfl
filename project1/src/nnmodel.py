import numpy as np

import torch
import torch.nn as nn
from torch import optim

from mlmodel import MLModel

from bci_dataset import BCIDataSet

from torchvision import transforms

from preprocessing import Normalize
from data_augmentation import GaussianNoise
from data_augmentation import Crop1d


class NNModel(MLModel, nn.Module):

    def update_data(self, data=None, targets=None, feature_augmentation=None):
        MLModel.update_data(self, data, targets, feature_augmentation)

        if data is not None:
            self.normalizer = Normalize(torch.min(self.data), torch.max(self.data))
            self.train_transform = transforms.Compose([  # Crop1d(50),
                                                       GaussianNoise(0, 2),
                                                       self.normalizer])
            self.test_transform = transforms.Compose([  # Crop1d,
                                                      self.normalizer])
            self.data_set = BCIDataSet(self.data, targets, transform=self.train_transform)

    def __init__(self, data=None, targets=None, feature_augmentation=None):
        MLModel.__init__(self)
        nn.Module.__init__(self)
        self.normalizer = None
        self.train_transform = None
        self.test_transform = None
        self.data_set = None
        self.update_data(data=data, targets=targets, feature_augmentation=feature_augmentation)

    def forward(self, x):
        return

    def fit(self, data=None, targets=None, validation_data=None, validation_targets=None, epochs=30, batch_size=128,
            optimizer='adam', lr=0.01, momentum=0):
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

        history = dict(loss=[], acc=[], val_loss=[], val_acc=[])

        criterion = nn.CrossEntropyLoss()

        epoch = 0

        try:
            for epoch in range(epochs):  # loop over the data set multiple times
                running_loss = 0.0
                xxx = 0
                for i, batch in enumerate(data_loader):
                    xxx = xxx + 1
                    # get the inputs
                    inputs, labels = batch

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self(inputs)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                test = 0
                # for i, x in enumerate(data):
                test += criterion(self(data[0:1]), targets[0:1]).item()
                running_loss = running_loss/len(data_loader)
                accuracy = self.compute_accuracy(data, targets)
                history['loss'].append(running_loss)
                history['acc'].append(accuracy)

                if validation_data is not None and validation_targets is not None:
                    val_acc = self.compute_accuracy(validation_data, validation_targets)
                    history['val_acc'].append(val_acc)
                    criterion2 = nn.CrossEntropyLoss()
                    val_loss = criterion2(self.predict(validation_data, raw=True), validation_targets).item()

                    val_loss = val_loss  # /validation_data.shape[0]
                    history['val_loss'].append(val_loss)

                self.train()
        except KeyboardInterrupt:
            print('Interrupted at epoch {}.'.format(epoch))
        return history

    def predict(self, data, raw=False):
        self.eval()
        # data = data.view(data.shape[0], 1, -1)
        data = self.test_transform(data)
        outputs = self(data)
        if not raw:
            _, outputs = torch.max(outputs.data, dim=1)
        return outputs

    def compute_accuracy(self, x_test, y_test):
        y_hat = self.predict(x_test).numpy()
        y_test = y_test.numpy()
        return np.sum(y_test == y_hat)/np.size(y_test)
