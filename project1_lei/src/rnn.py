import torch
import torch.nn as nn
from torch.autograd import Variable
import dlc_bci as bci
import torch.utils.data
from torch import Tensor
import numpy as np
import torch.nn.functional as F
from torch import optim


# Hyper Parameters
sequence_length = 28
input_size = 50
hidden_size =10
num_layers = 2
num_classes = 2
batch_size = 79
learning_rate = 0.01

def data_augmentation(data,method):
    data_transfor = Tensor(data.shape[0], data.shape[1], data.shape[2])
    if method =="noise":
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                #wn = Tensor(np.random.randn(len(data[i,j,:])))
                #data_transfor[i,j,:] = data[i,j,:] +10*wn
                data_transfor[i,j,:] = data[i,j,:]+Tensor(np.random.normal(0, 1,data[i,j,:].shape))
    if method=="roll":
        data_transfor = Tensor(np.roll(data,15,axis=2))
    # if method =="stretch":
    #     rate = 0.6
    #     input_length = 50
    #     for i in range(data.shape[0]):
    #         for j in range(data.shape[1]):
    #             data1 = librosa.effects.time_stretch(np.array(data[i,j,:]), rate)
    #             if len(data1) > input_length:
    #                 data1 = data1[:input_length]
    #             else:
    #                 data1 = np.pad(data1, (0, max(0, input_length - len(data1))), "constant")
    #             data_transfor[i,j,:] = Tensor(data1)
    return torch.cat([data,data_transfor],0)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        #self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out


rnn = RNN(input_size, hidden_size, num_layers, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)




train_input, train_target = bci.load(root='../data',one_khz=False)
test_input,test_target = bci.load(root='../data',one_khz=False,train=False)
print(str(type(train_input)), train_input.size())
print(str(type(train_target)), train_target.size())
# mean,std = torch.mean(train_input,0),torch.std(train_input,0)
# train_input.sub_(mean).div_(std)
# test_input.sub_(mean).div_(std)
# train_input = data_augmentation(train_input,"roll")
# train_target = torch.cat([train_target,train_target],0)


mean,std = torch.mean(train_input,0),torch.std(train_input,0)
train_input.sub_(mean).div_(std)
test_input.sub_(mean).div_(std)
train = list((zip(train_input,train_target)))
test = list((zip(test_input,test_target)))
train_loader = torch.utils.data.DataLoader(dataset=train,batch_size = 79,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test,batch_size = 20,shuffle=True)
# train_input = Variable(train_input)
# train_target = Variable(train_target)
# test_input = Variable(test_input)
# test_target = Variable(test_target)



num_epochslist = [200,15,15,15,15,15]
for num_epochs in num_epochslist:

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, sequence_length, input_size))
            labels = Variable(labels)
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = rnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 2 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                      % (epoch + 1, num_epochs, i + 1, len(train_input) // batch_size, loss.data[0]))
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, sequence_length, input_size))
        outputs = rnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('EPOCH:',num_epochs,'Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))