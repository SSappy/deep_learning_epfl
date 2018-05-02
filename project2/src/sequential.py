from torch import FloatTensor

from random import shuffle

class Sequential:
    def __init__(self, loss, input_size):
        """
        Initialize a sequential model (succession of layers).
        :param loss: object with methods compute_loss and compute_grad (cf. losses.py)
        :param input_size: size of input samples of the network
        """
        self.loss = loss
        self.input_size = input_size
        self.layers = []

    def check_consistency(self, layer):
        """
        Check if layer's input size is consistent with hidden size of current last layer
        :param layer: layer to test
        :return: boolean depending on result of test
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

    def add_layer(self, layer):
        """
        Add a new layer to the model
        :param layer: layer to be added (instance of an object implementing Module interface)
        """
        if self.check_consistency(layer):
            self.layers.append(layer)

    def forward(self, model_input):
        """
        Forward propagation of an input sample through the network.
        :param model_input: input sample to be propagated through the network.
        :return: output of the last layer after propagation of the input sample.
        """
        for layer in self.layers:
            model_input = layer.forward(model_input)
        return model_input

    def backward(self, grad_wrt_output, step_size):
        """
        Proceed to a backward pass on the network (and weight updates)
        :param grad_wrt_output: gradient with respect to the output
        :param step_size: step size for the weights and bias updates
        """
        for layer in reversed(self.layers):
            grad_wrt_output = layer.backward(grad_wrt_output)

    def gradient_step(self, step_size):
        for layer in self.layers:
            layer.gradient_step(step_size)

    def fit(self, x_train, y_train, x_validation, y_validation, epochs=100, step_size=0.1):
        """
        Fit the network.
        :param x_train: training samples (Tensor of shape (num_samples, self.input_size))
        :param y_train: targets of the training samples (Tensor of shape (num_samples, output_size)
        :param x_validation: validation samples (Tensor of shape (num_samples, self.input_size))
        :param y_validation: targets of the validation samples (Tensor of shape (num_samples, output_size)
        :param epochs: number of epochs for the training
        :param step_size: step size of the weights and bias updates
        :return: training and validation accuracies
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
                # Forward-pass
                output = self.forward(x_train[i])

                # Backward-pass
                grad_wrt_output = self.loss.compute_grad(output, y_train[i])
                self.backward(grad_wrt_output, step_size=step_size)

                # Gradient step
                self.gradient_step(step_size)

            step_size = step_size * 0.9

            _, loss = self.predict(x_train, y_train)
            print('Loss at epoch {} : {}'.format(epoch, loss.mean()))

        # Compute and print training performance
        training_predictions, training_loss = self.predict(x_train, y_train)
        training_accuracy = (training_predictions == y_train).sum() / y_train.shape[1] / training_predictions.shape[0]
        print('\nTraining loss : {}'.format(training_loss.mean()))
        print('Training accuracy : {}\n'.format(training_accuracy))

        # Compute and print validation performance
        validation_predictions, validation_loss = self.predict(x_validation, y_validation)
        validation_accuracy = (validation_predictions == y_validation).sum() / y_validation.shape[1] / validation_predictions.shape[0]
        print('Validation loss : {}'.format(validation_loss.mean()))
        print('Validation accuracy : {}'.format(validation_accuracy))

        return training_accuracy, validation_accuracy


    def predict(self, x, y=None):
        """
        Make a prediction using the trained network.
        :param x: input samples
        :param y: output targets, usually None except when called during training (fit method)
        :return: predictions (and loss if called within fit method)
        """
        output_size = self.layers[-1].get_hidden_size()
        predictions = FloatTensor(x.shape[0], output_size).zero_()

        for i in range(x.shape[0]):
            predictions[i] = self.forward(x[i])

        if y is not None:
            loss = self.loss.compute_loss(predictions, y)
            _, ind = predictions.max(1)
            predictions = FloatTensor(predictions.shape).zero_().scatter_(1, ind.view(-1, 1), 1)
            return predictions, loss

        else:
            _, ind = predictions.max(1)
            predictions = FloatTensor(predictions.shape).zero_().scatter_(1, ind.view(-1, 1), 1)
            return predictions

    def summary(self):
        print('Model with {} layers'.format(len(self.layers)))
        print('\tInput size : {}'.format(self.input_size))
        for layer in self.layers[:-1]:
            layer.summary()
        print('\t{} fully connected output units'.format(self.layers[-1].get_hidden_size()))