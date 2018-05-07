import torch
import torch.nn as nn
from torch.nn import functional as F

from nnmodel import NNModel


# These are just tests at the moment

class BasicLSTM(NNModel):

    def __init__(self, input_size, hidden_size):
        super(BasicLSTM, self).__init__()
        self.hidden_size = hidden_size

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_size, 2)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(2, 16, self.hidden_size),
                torch.zeros(2, 16, self.hidden_size))

    def forward(self, x):
        print('x ', x.shape)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        print('lstm out ', lstm_out.shape)
        tag_space = self.hidden2tag(x.view(len(x), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


class RNN(NNModel):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        # self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        # Set initial states
        print('x ', x.shape)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        print('h0 ', h0.shape)

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))

        # Decode hidden state of last time step
        print('out[:, -1, :] ', out[:, -1, :].shape)
        out = self.fc(out[:, -1, :])
        return out
