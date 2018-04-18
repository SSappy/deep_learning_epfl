from torch import FloatTensor, LongTensor, mul
from numpy import tanh as nptanh # TODO implement tanh function
from module import Module


class ReLU(Module):

    def __init__(self, input_size):
        self.hidden_size = input_size
        self.input = FloatTensor(input_size)
        self.output = FloatTensor(input_size)
        self.grad_wrt_input = FloatTensor(input_size)

    def forward(self, input_tensor):
        """
        :param input_tensor: tensor of shape (self.size)
        :return: tensor of shape (self.size) which is the result of applying element_wise
        ReLU function to the input_tensor
        """
        self.output = input_tensor
        self.output[input_tensor < 0] = 0
        return self.output

    def backward(self, grad_wrt_output, step_size=None):
        """
        :param grad_wrt_output: tensor of shape (self.size) which is the gradient with respect
        to the output of the current layer
        :return: tensor of shape (self.size) which is the element-wise product of gradient of ReLU and grad_wrt_output.
        """
        derivative = self.output
        derivative[derivative != 0] = 1
        self.grad_wrt_input = grad_wrt_output * derivative
        return self.grad_wrt_input

    def param(self):
        return []

    def get_hidden_size(self):
        return self.hidden_size

    def get_input_size(self):
        return self.hidden_size


class tanh(Module):

    def __init__(self, input_size):
        self.hidden_size = input_size
        self.input = FloatTensor(input_size)
        self.output = FloatTensor(input_size)
        self.grad_wrt_input = FloatTensor(input_size)

    def forward(self, input_tensor):
        """
        :param input_tensor: tensor of shape (self.size)
        :return: tensor of shape (self.size) which is the result of applying element_wise
        tanh function to the input_tensor
        """
        self.output = nptanh(input_tensor)
        return self.output

    def backward(self, grad_wrt_output, step_size=None):
        """
        :param grad_wrt_output: tensor of shape (self.size) which is the gradient with respect
        to the output of the current layer
        :return: tensor of shape (self.size) which is the element-wise product of gradient of tanh and grad_wrt_output.
        """
        derivative = 1 - mul(self.output, self.output)
        self.grad_wrt_input = grad_wrt_output * derivative
        return self.grad_wrt_input

    def param(self):
        return []

    def get_hidden_size(self):
        return self.hidden_size

    def get_input_size(self):
        return self.hidden_size
