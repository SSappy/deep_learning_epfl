import torch.nn as nn
from torch.nn import functional as F

from nnmodel import NNModel


class LSTMNet(NNModel):

    def __init__(self, dim_input, dim_recurrent, num_layers, dim_output):
        super(LSTMNet, self).__init__()
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
