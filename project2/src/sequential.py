from numpy.random import shuffle
from numpy import ndarray

class Sequential:
    def __init__(self, loss, input_size, output_size):
        self.loss = loss
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []

    def check_consistency(self, layer):
        """

        :param layer:
        :return:
        """
        if len(self.layers) == 0 and layer.get_input_size() == self.input_size:
            return True
        elif len(self.layers) > 0:
            if self.layers[-1].get_hidden_size() == layer.get_input_size():
                return True
        else:
            print('Mismatch in the shapes.')
            print('Trying to append layer of size [{}] to layer of size [{}]').format(layer.get_input_size(),
                                                                                      self.layers[-1].get_hidden_size())
            return False

    def add(self, layer):
        if self.check_consistency(layer):
            self.layers.append(layer)

    def forward(self, model_input):
        for layer in self.layers:
            model_input = layer.forward(model_input)
        return model_input

    def backward(self, grad_wrt_output, step_size):
        for layer in reversed(self.layers):
            grad_wrt_output = layer.backward(grad_wrt_output, step_size=step_size)


    def fit(self, x_train, y_train, x_validation, y_validation, epochs=100, step_size=0.0001):
        """

        :param x_train:
        :param y_train:
        :param x_validation:
        :param y_validation:
        :param epochs:
        :param step_size:
        :return:
        """
        try:
            assert x_train.shape[1] == self.input_size
        except AssertionError:
            print('Wrong input shape, samples should have shape [{}] but received [{}]'.format(self.input_size,
                                                                                               x_train.shape[1]))
            return

        for epoch in range(1, epochs+1):
            # shuffle indexes in order for GD to look at samples in random order
            idx = list(range(x_train.shape[0]))
            shuffle(idx)

            for i in idx:
                # forward-pass
                output = self.forward(x_train[i])

                # backward-pass
                grad_wrt_output = self.loss.compute_grad(output, y_train[i])
                self.backward(grad_wrt_output, step_size=step_size)

        # Compute and print validation performance
        predictions, validation_loss = self.predict(x_validation, y_validation)
        print('Validation loss : {}'.format(validation_loss))
        print('Validation accuracy : {}'.format((predictions == y_validation).sum()/predictions.shape[0]))


    def predict(self, x, y=None):
        """

        :param x:
        :param y:
        :return:
        """
        predictions = ndarray(shape=(x.shape[0], self.output_size))

        for i in range(x.shape[0]):
            predictions[i] = self.forward(x[i])
        loss = self.loss.compute_loss(predictions, y)

        predictions[predictions.squeeze() > 0.5] = 1
        predictions[predictions.squeeze() <= 0.5] = 0

        if y is not None:
            return predictions, loss
        else:
            return predictions
