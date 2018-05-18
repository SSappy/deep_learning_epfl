"""
File defining the recurrent neural network models that we trained. They all inherit the class NNModel defined in
nnmodel.py.
"""

import torch.nn as nn
from torch.nn import functional as F

from nnmodel import NNModel


class LSTMNet1(NNModel):

    def __init__(self, dim_input, dim_recurrent, num_layers, dim_output):
        super(LSTMNet1, self).__init__()
        self.lstm = nn.LSTM(input_size=dim_input, hidden_size=dim_recurrent, num_layers=num_layers, batch_first=True)
        self.fc_o2y = nn.Linear(dim_recurrent, dim_output)

    def forward(self, x):
        # Makes  this a batch  of size 1
        # The  first  index  is the time , sequence  number  is the  second
        x = x.view(x.shape[0], -1)
        x = x.unsqueeze(1)
        # Get the  activations  of all  layers  at the  last  time  step
        output, _ = self.lstm(x)
        # Drop  the  batch  index
        output = output.squeeze(1)
        return self.fc_o2y(F.relu(output))


class LSTMNet1Dropout(NNModel):

    def __init__(self, dim_input, dim_recurrent, num_layers, dim_output, dropout=0.5):
        super(LSTMNet1Dropout, self).__init__()
        self.lstm = nn.LSTM(input_size=dim_input, hidden_size=dim_recurrent, num_layers=num_layers, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc_o2y = nn.Linear(dim_recurrent, dim_output)

    def forward(self, x):
        # Makes  this a batch  of size 1
        # The  first  index  is the time , sequence  number  is the  second
        x = x.view(x.shape[0], -1)
        x = x.unsqueeze(1)
        # Get the  activations  of all  layers  at the  last  time  step
        output, _ = self.lstm(x)
        # Drop  the  batch  index
        output = output.squeeze(1)
        output = self.drop(output)
        return self.fc_o2y(F.relu(output))


class ConvLSTM(NNModel):

    def __init__(self, dropout=0.5):
        super(ConvLSTM, self).__init__()
        self.conv = nn.Conv1d(28, 32, kernel_size=5, padding=2)
        self.drop1 = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=32*25, hidden_size=128, num_layers=1, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc_o2y = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv(x), kernel_size=2))
        x = self.drop1(x)
        # Makes  this a batch  of size 1
        # The  first  index  is the time , sequence  number  is the  second
        x = x.view(x.shape[0], -1)
        x = x.unsqueeze(1)
        # Get the  activations  of all  layers  at the  last  time  step
        output, _ = self.lstm(x)
        # Drop  the  batch  index
        output = output.squeeze(1)
        output = self.drop2(output)
        return self.fc_o2y(F.relu(output))
