from torch import FloatTensor, matmul

from module import Module

class Linear(Module):
    def __init__(self, input_size, hidden_units):
        """
        :param input_size: size of the input tensor
        :param hidden_units: number of hidden units in the layer
        Several objects are created among which :
            - a tensor weights of shape (hidden_units, input_size) where the i-th row is the i-th weight vector w_i
            - a tensor biases of shape (hidden_units) where b_i is the i-th bias
        """
        self.input_size = input_size
        self.hidden_size = hidden_units
        self.input = FloatTensor(input_size)
        self.output = FloatTensor(hidden_units)

        self.weights = FloatTensor(hidden_units, input_size).uniform_(-1, 1)
        self.biases = FloatTensor(hidden_units).uniform_(-1, 1)

        self.weights_gradients = FloatTensor(hidden_units, input_size)
        self.biases_gradients = FloatTensor(hidden_units)


    def forward(self, input_tensor):
        """
        Forward propagation of input through the layer.
        :param input_tensor: torch Tensor of shape self.input_size
        :return: output_tensor of shape (self.hidden_size) which is the result of WX + b
        """
        self.input = input_tensor
        self.output = matmul(self.weights, input_tensor) + self.biases
        return self.output

    def backward(self, grad_wrt_output, step_size=0.1):
        """
        Backward pass, including weights and biases updates.
        Gradients are computed using the gradient with respect to the output of the layer
        :param grad_wrt_output: gradient with respect to the output
        :param step_size: step size of the updates
        :return: gradient with respect to the input (to be passed up the network)
        """
        # computing gradients
        self.biases_gradients = grad_wrt_output
        x_transpose = self.input.clone().resize_(1, self.input.size()[0])
        self.weights_gradients = matmul(grad_wrt_output.resize_(grad_wrt_output.size()[0], 1), x_transpose)

        # computing gradient with respect to input
        grad_wrt_input = matmul(self.weights.transpose(0, 1), grad_wrt_output)

        # updating weights and biases
        self.weights -= step_size * self.weights_gradients
        self.biases -= step_size * self.biases_gradients

        return grad_wrt_input


    def param(self):
        """
        :return: A list of pairs, each composed of a parameter tensor, and a gradient tensor of same size.
        """
        return [(self.weights[i, :], self.weights_gradients[i, :]) for i in range(self.hidden_size)] \
               + [(self.biases, self.biases_gradients)]

    def get_hidden_size(self):
        """
        :return: hidden size of the network
        """
        return self.hidden_size

    def get_input_size(self):
        """
        :return: input size of the network
        """
        return self.input_size

    def summary(self):
        print('\tFully connected layer of {} hidden units'.format(self.hidden_size))