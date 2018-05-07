import torch
import torch.nn as nn
from torch.nn import functional as F

from nnmodel import NNModel


# These are just tests at the moment

class BasicLSTM(NNModel):

    def __init__(self, input_size, hidden_size):
        super(LSTMTagger, self).__init__()
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
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        tag_space = self.hidden2tag(x.view(len(x), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
