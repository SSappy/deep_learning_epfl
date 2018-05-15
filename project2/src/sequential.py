from torch import FloatTensor
from torch import cat

from collections import defaultdict
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

    def backward(self, grad_wrt_output, momentum=None):
        """
        Proceed to a backward pass on the network (computation of the gradients)
        :param grad_wrt_output: gradient with respect to the output
        :param momentum: if not None, Momentum SGD is used with the value passed
        """
        for layer in reversed(self.layers):
            grad_wrt_output = layer.backward(grad_wrt_output, momentum)

    def gradient_step(self, step_size):
        """
        Proceed to a gradient step on the network (update of the weights and biases computed during the backward pass).
        :param step_size: step size for the weights and bias updates
        """
        for layer in self.layers:
            layer.gradient_step(step_size)

    def fit(self, x_train, y_train, x_validation, y_validation, epochs=100,
            batch_size=1, step_size=0.1, momentum=None):
        """
        Fit the network.
        :param x_train: training samples (Tensor of shape (num_samples, self.input_size))
        :param y_train: targets of the training samples (Tensor of shape (num_samples, output_size)
        :param x_validation: validation samples (Tensor of shape (num_samples, self.input_size))
        :param y_validation: targets of the validation samples (Tensor of shape (num_samples, output_size)
        :param epochs: number of epochs for the training
        :param batch_size: batch size for the batch SGD algorithm
        :param step_size: step size of the weights and bias updates
        :param momentum: if not None, Momentum SGD is used with the value passed
        :return: history (dictionary containing validation and training losses and accuracies for all epochs)
        """
        try:
            assert x_train.shape[1] == self.input_size
        except AssertionError:
            print('Wrong input shape, samples should have shape [{}] but received [{}]'.format(self.input_size,
                                                                                               x_train.shape[1]))
            return

        history = defaultdict(list)

        for epoch in range(1, epochs+1):
            # shuffle indexes in order for GD to look at samples in random order
            idx = list(range(x_train.shape[0]))
            shuffle(idx)

            batches = [idx[i:i+batch_size] for i in range(0, len(idx), batch_size)]
            for batch in batches:
                # Forward-pass
                outputs = FloatTensor()
                targets = FloatTensor()
                for i in batch:
                    output = self.forward(x_train[i])
                    outputs = cat((outputs, output.view(1, -1)), 0)
                    targets = cat((targets, y_train[i].view(1, -1)), 0)

                # Backward-pass
                grad_wrt_output = self.loss.compute_grad(outputs, targets)
                self.backward(grad_wrt_output, momentum)

                # Gradient step
                self.gradient_step(step_size)

            step_size = step_size * 0.9

            tr_predictions, tr_loss = self.predict(x_train, y_train)
            tr_accuracy = (tr_predictions == y_train).sum() / y_train.shape[1] / tr_predictions.shape[0]
            history['tr_loss'].append(tr_loss.mean())
            history['tr_acc'].append(tr_accuracy)

            val_predictions, val_loss = self.predict(x_validation, y_validation)
            val_accuracy = (val_predictions == y_validation).sum() / y_validation.shape[1] / val_predictions.shape[0]
            history['val_loss'].append(val_loss.mean())
            history['val_acc'].append(val_accuracy)

            print('Loss at epoch {} : {}'.format(epoch, tr_loss.mean()))

        # Print final training performance
        print('\nTraining loss : {}'.format(history['tr_loss'][-1]))
        print('Training accuracy : {}\n'.format(history['tr_acc'][-1]))

        # Print final validation performance
        print('Validation loss : {}'.format(history['val_loss'][-1]))
        print('Validation accuracy : {}'.format(history['val_acc'][-1]))

        return history


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