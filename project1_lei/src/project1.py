import dlc_bci as bci
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
import math
from scipy.ndimage import gaussian_filter1d
from torch.autograd import Variable
from torch import nn
from torch import optim
from torch.nn import functional as F
from sklearn.model_selection import train_test_split

# def train_model(model,train_input,train_target,mini_batch_size,learning_rate):
#     optimizer = optim.Adadelta(model.parameters(),lr = learning_rate)
#     iteration = 500
#     Loss = []
#     accuracy_list=[]
#     Accuracy = 0
#     count = 0
#     index_random = np.arange(train_target.shape[0])
#     np.random.shuffle(index_random)
#     indexes = index_random.tolist()
#     train_target_shuffle = train_target[indexes]
#     train_input_shuffle = train_input[indexes]
#     size = int(train_target.shape[0] * 0.75)
#     X_train = train_input_shuffle[:size, :, :]
#     X_test = train_input_shuffle[size:, :, :]
#     y_train = train_target_shuffle[:size]
#     y_test = train_target_shuffle[size:]
#     for i in range(iteration):
#         sum_loss = 0
#         for n in range(0,X_train.shape[0],mini_batch_size):
#             output = model(X_train.narrow(0,n,mini_batch_size))
#             loss = F.cross_entropy(output,y_train.narrow(0,n,mini_batch_size))
#             model.zero_grad()
#             loss.backward()
#             sum_loss = loss.data[0] + sum_loss
#             Loss.append(sum_loss)
#             optimizer.step()
#         print(i, 'loss= ', sum_loss)
#         accuracy = 1 - compute_nb_error(model,X_test,y_test,mini_batch_size)/X_test.shape[0]
#         if accuracy>=Accuracy:
#             Accuracy = accuracy
#             count=0
#         elif(accuracy<Accuracy-0.02):
#             count+=1
#             if count==20:
#                 break
#         print('accuracy=',accuracy,'Accuracy',Accuracy)
#         accuracy_list.append(accuracy)
#     return Loss,accuracy_list


##-----------original one --------##
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        nb_hidden = 32
        self.conv1 = nn.Conv1d(28, 32, kernel_size=5,padding=2)
        #self.batch_nom1= nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3,padding=1)
        self.batch_nom2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3,padding=1)
        self.conv3_drop = nn.Dropout()
        self.batch_nom3 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(12 * 32, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv1(x), kernel_size=2))
        x = F.relu(self.batch_nom2(F.max_pool1d(self.conv2(x), kernel_size=2)))
        x = F.relu(self.batch_nom3(self.conv3_drop(self.conv3(x))))
        x = F.relu(self.fc1(x.view(-1, 12*32)))
        #x = F.dropout(x,training=self.training)
        x = self.fc2(x)
        return x

##--combine fft and time series signal as 2D signal ----#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         nb_hidden = 200
#         self.conv1 = nn.Conv2d(28, 32, kernel_size=5,padding=2)
#         #self.batch_nom1= nn.BatchNorm1d(32)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3,padding=1)
#         # self.batch_nom2 = nn.BatchNorm1d(32)
#         #self.conv3 = nn.Conv2d(32, 64, kernel_size=3,padding=1)
#         #self.conv3_drop = nn.Dropout()
#         #self.batch_nom3 = nn.BatchNorm1d(64)
#         self.fc1 = nn.Linear(144 * 64, nb_hidden)
#         self.fc2 = nn.Linear(nb_hidden, 2)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2,stride=2))
#         x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2,stride=2))
#         #x = F.relu(self.conv3(x))
#         x = F.relu(self.fc1(x.view(-1, 144*64)))
#         #x = F.dropout(x,training=self.training)
#         x = self.fc2(x)
#         return x

def train_model(model,train_input,train_target,mini_batch_size,learning_rate):
    optimizer = optim.Adam(model.parameters(),lr = learning_rate)
    #criterion = nn.MSELoss()
    iteration = 500
    Loss = []
    for i in range(iteration):
        sum_loss = 0
        for n in range(0,train_input.shape[0],mini_batch_size):
            output = model(train_input.narrow(0,n,mini_batch_size))
            loss = F.cross_entropy(output,train_target.narrow(0,n,mini_batch_size))
            #loss = criterion(output, train_target.narrow(0, n, mini_batch_size))
            model.zero_grad()
            loss.backward()
            sum_loss = loss.data[0] + sum_loss
            Loss.append(sum_loss)
            optimizer.step()
        print(i,' ',sum_loss)
    return Loss

def compute_nb_error(model,data_input,target,mini_batch_size):
    error = 0
    for n in range(0,data_input.shape[0],mini_batch_size):
        output = model(data_input.narrow(0,n,mini_batch_size))
        _,prediction = output.data.max(1)
        for k in range(mini_batch_size):
            if target.data[n+k]!= prediction[k]:
                error +=1
    return error

# def preprocessing(data,size,sigma):
#     mean,std = torch.mean(data,0),torch.std(data,0)
#     data.sub_(mean).div_(std)
#     # filter = gaussian_filter1d(np.arange(-size,size+1), sigma)
#     # for i in range(data.shape[0]):
#     #      for j in range(data.shape[1]):
#     #          data[i,j,:] = Tensor(np.convolve(data[i,j,:],filter,'same'))
#     return data

def data_augmentation(data,method):
    data_transfor = Tensor(data.shape[0], data.shape[1], data.shape[2])
    if method =="noise":
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                #wn = Tensor(np.random.randn(len(data[i,j,:])))
                #data_transfor[i,j,:] = data[i,j,:] +10*wn
                data_transfor[i,j,:] = data[i,j,:]+Tensor(np.random.normal(0, 1,data[i,j,:].shape))
    if method=="roll":
        data_transfor = Tensor(np.roll(data,10,axis=2))
    if method =="stretch":
        rate = 0.6
        input_length = 50
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data1 = librosa.effects.time_stretch(np.array(data[i,j,:]), rate)
                if len(data1) > input_length:
                    data1 = data1[:input_length]
                else:
                    data1 = np.pad(data1, (0, max(0, input_length - len(data1))), "constant")
                data_transfor[i,j,:] = Tensor(data1)
    return torch.cat([data,data_transfor],0)

# def preprocessing(data):
#     A = Tensor(data.shape[0],data.shape[1],data.shape[2])
#     B = Tensor(data.shape[0],data.shape[1],data.shape[2],data.shape[2])
#     for i in range(data.shape[0]):
#         for j in range(data.shape[1]):
#             A[i,j,:] = Tensor(np.fft.fft(data[i,j,:]))
#             for z in range(data.shape[2]):
#                 B[i,j,z,:] = data[i,j,:]
#                 B[i,j,:,z] = A[i,j,:]
#     return B

train_input, train_target = bci.load(root='../data',one_khz=False)
test_input,test_target = bci.load(root='../data',one_khz=False,train=False)
print(str(type(train_input)), train_input.size())
print(str(type(train_target)), train_target.size())


# train_input = data_augmentation(train_input,"stretch")
#train_input = data_augmentation(train_input,"roll")
# train_target = torch.cat([train_target,train_target],0)
#train_target = torch.cat([train_target,train_target],0)
mean,std = torch.mean(train_input,0),torch.std(train_input,0)
train_input.sub_(mean).div_(std)
#train_input = preprocessing(train_input)

# index_random = np.arange(train_target.shape[0])
# np.random.shuffle(index_random)
# indexes = index_random.tolist()
# train_input = train_input[indexes]
# train_target = train_target[indexes]
test_input.sub_(mean).div_(std)
#test_input = preprocessing(test_input)



train_input = Variable(train_input)
train_target = Variable(train_target)
test_input = Variable(test_input)
test_target = Variable(test_target)
mini_batch_size = 79
learning_rate = 0.001
model = Net()
Loss = train_model(model,train_input,train_target,mini_batch_size,learning_rate)
plt.plot(Loss)
plt.show()
error = compute_nb_error(model,train_input,train_target,79)
print(100*(1-error/train_target.shape[0]))
error = compute_nb_error(model,test_input,test_target,20)
print(100*(1-error/test_target.shape[0]))