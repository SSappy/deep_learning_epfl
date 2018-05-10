import torch.nn as nn
from torch.nn import functional as F

from nnmodel import NNModel


class ConvNet1(NNModel):

    def __init__(self, one_khz=False):
        super(ConvNet1, self).__init__()
        num_hidden = 32
        self.length = 25
        if one_khz:
            self.length = 250
        self.conv1 = nn.Conv1d(28, 32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(self.length*32, num_hidden)
        self.fc2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv1(x), kernel_size=2))
        x = F.relu(self.fc1(x.view(-1, self.length*32)))
        x = self.fc2(x)
        return x


class ConvNet1Dropout(NNModel):

    def __init__(self, dropout=0.5):
        super(ConvNet1Dropout, self).__init__()
        num_hidden = 32
        self.length = 25
        self.conv1 = nn.Conv1d(28, 32, kernel_size=5, padding=2)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.length*32, num_hidden)
        self.fc2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.drop(self.conv1(x)), kernel_size=2))
        x = F.relu(self.fc1(x.view(-1, self.length*32)))
        x = self.fc2(x)
        return x


class ConvNet1DropoutCrop42(NNModel):

    def __init__(self, dropout=0.5):
        super(ConvNet1DropoutCrop42, self).__init__()
        num_hidden = 32
        self.length = 21
        self.conv1 = nn.Conv1d(28, 32, kernel_size=5, padding=2)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.length*32, num_hidden)
        self.fc2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.drop(self.conv1(x)), kernel_size=2))
        x = F.relu(self.fc1(x.view(-1, self.length*32)))
        x = self.fc2(x)
        return x


class ConvNet2(NNModel):

    def __init__(self, one_khz):
        super(ConvNet2, self).__init__()
        num_hidden = 64
        self.length = 12
        if one_khz:
            self.length = 125
        self.conv1 = nn.Conv1d(28, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(self.length*64, num_hidden)
        self.fc2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool1d(self.conv2(x), kernel_size=2))
        x = F.relu(self.fc1(x.view(-1, self.length*64)))
        x = F.softmax(self.fc2(x), dim=0)
        return x


class TestNet(NNModel):

    def __init__(self):
        super(TestNet, self).__init__()
        nb_hidden = 32
        self.conv1 = nn.Conv1d(28, 32, kernel_size=5, padding=2)
        # self.batch_nom1= nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.batch_nom2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.conv3_drop = nn.Dropout()
        self.batch_nom3 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(12 * 32, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv1(x), kernel_size=2))
        x = F.relu(self.batch_nom2(F.max_pool1d(self.conv2(x), kernel_size=2)))
        x = F.relu(self.batch_nom3(self.conv3_drop(self.conv3(x))))
        x = F.relu(self.fc1(x.view(-1, 12 * 32)))
        # x = F.dropout(x,training=self.training)
        x = self.fc2(x)
        return x
