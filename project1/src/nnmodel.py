"""
File defining the class NNModel, inheriting the class MLModel. It is the base class for all neural network models.
"""


import numpy as np

import torch
import torch.nn as nn
from torch import optim

from mlmodel import MLModel

from utils.bci_dataset import BCIDataSet

from torchvision import transforms

from utils.preprocessing import Normalize
from utils.preprocessing import Standardize
from utils.data_augmentation import GaussianNoise
from utils.data_augmentation import Crop1d


class NNModel(MLModel, nn.Module):

    def update_data(self, data=None, targets=None, feature_augmentation=None, standardize=False,
                    normalize=False, noise=None, crop=None):
        """
        Update the data set used to train the model.
        :param data: Raw data (transformations are made automatically).
        :param targets: Labels of the data.
        :param feature_augmentation: A feature augmentation function that may be applied to the data set.
        :param standardize: If True standardize the data.
        :param normalize: If True normalize the data.
        :param noise: Add a gaussian noise of mean 0 and std noise if not None.
        :param crop: Crop a random window in the data of size crop if not None.
        :return: Nothing.
        """
        MLModel.update_data(self, data, targets, feature_augmentation)

        if data is not None:
            self.train_transform = []
            self.test_transform = []

            self.crop = crop
            if crop:
                self.train_transform.append(Crop1d(crop))
                self.test_transform.append(Crop1d(crop))

            if noise is not None:
                self.train_transform.append(GaussianNoise(0, noise))

            if standardize:
                self.normalizer = Standardize(torch.mean(self.data), torch.std(self.data))
            elif normalize:
                self.normalizer = Normalize(torch.min(self.data), torch.max(self.data))
            else:
                self.normalizer = None

            if self.normalizer is not None:
                self.train_transform.append(self.normalizer)
                self.test_transform.append(self.normalizer)

            self.train_transform = transforms.Compose(self.train_transform)
            self.test_transform = transforms.Compose(self.test_transform)

            self.data_set = BCIDataSet(self.data, targets, transform=self.train_transform)

    def __init__(self, data=None, targets=None, feature_augmentation=None, **kwargs):
        """
        Initializer.
        :param data: Raw data set.
        :param targets: Labels of the data.
        :param feature_augmentation: Feature augmentation function applied to the data.
        """
        MLModel.__init__(self)
        nn.Module.__init__(self)
        self.crop = None
        self.normalizer = None
        self.train_transform = None
        self.test_transform = None
        self.data_set = None
        self.update_data(data=data, targets=targets, feature_augmentation=feature_augmentation, **kwargs)

    def forward(self, x):
        """
        This method has to be overridden by the sub classes of NNModel.
        :param x: Data point.
        :return: The output of the model.
        """
        return

    def fit(self, data=None, targets=None, validation_data=None, validation_targets=None, epochs=30, batch_size=16,
            optimizer='adam', lr=0.01, lr_decay=(0, 0), momentum=0, **kwargs):
        """
        Method used to fit the model to some data and targets.
        :param data: Raw data set.
        :param targets: Labels of the data.
        :param validation_data: Raw validation data.
        :param validation_targets: Labels of validation data.
        :param epochs: Number of epochs to train the model.
        :param batch_size: Size of the batches
        :param optimizer: Optimizer used : can be either Adam or SGD.
        :param lr: Learning rate for the optimizer.
        :param lr_decay: Tuple (step, gamma). If step is not null, multiply the learning rate by gamma every step
        epochs.
        :param momentum: Momentum for the SGD optimizer (if used).
        :return: An history of the loss and accuracy (and validation loss and accuracy if some validation
        data is given).
        """
        self.train()

        if isinstance(validation_data, float):
            perm = torch.randperm(data.shape[0])
            val_size = int(validation_data*data.shape[0])
            validation_data = data[perm[:val_size]]
            data = data[perm[val_size:]]
            validation_targets = targets[perm[:val_size]]
            targets = targets[perm[val_size:]]

        if isinstance(data, torch.utils.data.dataset.Dataset):
            data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                                      shuffle=True, num_workers=2)
        else:
            self.update_data(data=data, targets=targets, **kwargs)
            data_loader = torch.utils.data.DataLoader(self.data_set, batch_size=batch_size,
                                                      shuffle=True, num_workers=2)

        optimizer = optimizer.lower()
        if optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        else:
            raise ValueError('The argument "optimizer" is invalid')

        step_decay, gamma = lr_decay

        history = dict(loss=[], acc=[], val_loss=[], val_acc=[])

        criterion = nn.CrossEntropyLoss()

        epoch = 0

        try:
            for epoch in range(epochs):  # loop over the data set multiple times
                running_loss = 0.0
                for i, batch in enumerate(data_loader):
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
                running_loss = running_loss/len(data_loader)
                accuracy = self.compute_accuracy(data, targets)
                history['loss'].append(running_loss)
                history['acc'].append(accuracy)

                if validation_data is not None and validation_targets is not None:
                    val_acc = self.compute_accuracy(validation_data, validation_targets)
                    history['val_acc'].append(val_acc)

                    val_loss = criterion(self.predict(validation_data, raw=True), validation_targets).item()
                    val_loss = val_loss  # /validation_data.shape[0]
                    history['val_loss'].append(val_loss)

                self.train()

                if step_decay != 0 and epoch != 0 and epoch%step_decay == 0:
                    lr = gamma*lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
        except KeyboardInterrupt:
            print('Interrupted at epoch {}.'.format(epoch))
        return history

    def predict(self, data, poll=11, raw=False):
        """
        Method used to predict new data labels.
        :param data: Raw data.
        :param poll: In case the data is cropped, number of times it is cropped and predicted for the fina label poll.
        :param raw: Boolean specifying if the output should be one-hot encoded. Default : False.
        :return: Predicted labels.
        """
        self.eval()

        if self.crop:
            outputs = torch.Tensor()
            for _ in range(poll):
                transformed_data = self.test_transform(data)
                if raw:
                    if _ == 0:
                        outputs = self(transformed_data)
                    else:
                        outputs = outputs + self(transformed_data)
                else:
                    _, labels = torch.max(self(transformed_data), dim=1)
                    labels = labels.view(-1, 1)
                    outputs = torch.cat((outputs, labels.float()), 1)

            if raw:
                outputs = outputs/11
            else:
                outputs = outputs.mean(dim=1)
                outputs = outputs > 0.5
                outputs = outputs.long()
        else:
            data = self.test_transform(data)
            outputs = self(data)
            if not raw:
                _, outputs = torch.max(outputs.data, dim=1)
        return outputs

    def compute_accuracy(self, x_test, y_test):
        """
        Compute the accuracy of the model for some new data and new labels given.
        :param x_test: Raw data.
        :param y_test: Labels.
        :return: The accuracy of the model.
        """
        y_hat = self.predict(x_test).numpy()
        y_test = y_test.numpy()
        return np.sum(y_test == y_hat)/np.size(y_test)
