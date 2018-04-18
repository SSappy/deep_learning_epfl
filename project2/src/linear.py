from torch import FloatTensor, LongTensor, matmul
from module import Module
from numpy.random import normal


class Linear(Module):
    def __init__(self, input_size, hidden_units):
        """
        :param input_size: size of the input tensor
        :param hidden_units: number of hidden units in the layer
        Several objects are created:
            - a tensor weights of shape (hidden_units, input_size) where the i-th row is the i-th parameter w_i
            - a tensor biases of shape (hidden_units) where b_i is the i-th bias
        """
        self.input_size = input_size
        self.hidden_size = hidden_units
        self.input = FloatTensor(input_size)
        self.output = FloatTensor(hidden_units)

        self.weights = FloatTensor(normal(0, 1, size=(hidden_units, input_size)))
        self.biases = FloatTensor(normal(0, 1, size=hidden_units))

        self.weights_gradients = FloatTensor(hidden_units, input_size)
        self.biases_gradients = FloatTensor(hidden_units)


    def forward(self, input_tensor):
        """
        :param input_tensor:
        :return: output_tensor of shape (self.hidden_size) which is the result of WX + b
        """
        self.input = input_tensor
        self.output = matmul(self.weights, input_tensor) + self.biases
        return self.output

    def backward(self, grad_wrt_output, step_size=0.1):
        """

        :param grad_wrt_output:
        :param step_size:
        :return:
        """

        grad = grad_wrt_output.resize_(grad_wrt_output.size()[0], 1).expand(grad_wrt_output.size()[0], self.input_size)
        assert list(grad.size()) == [self.hidden_size, self.input_size]

        self.weights_gradients = grad * self.input
        self.biases_gradients = grad_wrt_output

        grad_wrt_input = matmul(self.weights.transpose(0, 1),grad_wrt_output)

        self.weights -= step_size * self.weights_gradients
        self.biases -= step_size * self.biases_gradients

        return grad_wrt_input


    def param(self):
        return [(self.weights[i, :], self.weights_gradients[i, :]) for i in range(self.hidden_size)] \
               + [(self.biases, self.biases_gradients)]

    def get_hidden_size(self):
        return self.hidden_size

    def get_input_size(self):
        return self.input_size