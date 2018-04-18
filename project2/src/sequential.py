from torch import FloatTensor, cat
from numpy import mean as npmean

class Sequential:
    def __init__(self, loss, input_size):
        self.loss = loss
        self.input_size = input_size
        self.layers = []

    def add(self, layer):
        if self.check_consistency(layer):
            self.layers.append(layer)
        else:
            # raise error
            print('error of shape')


    def fit(self, x_train, y_train, epochs=100, mini_batch_size=1):
        assert x_train.shape[1] == self.input_size

        num_batch = len(x_train) // mini_batch_size


        for epoch in range(1, epochs+1):
            errors = []
            predictions = FloatTensor()
            for k in range(num_batch-1): # TODO take the last samples if batch size not divider of number of samples
                current_batch = x_train[k*mini_batch_size : (k+1)*mini_batch_size]
                pred = FloatTensor()

                # forward pass
                for i in range(current_batch.shape[0]):
                    current = current_batch[i]

                    # go through all the layers
                    for layer in self.layers:
                        current = layer.forward(current)

                    # store the output
                    pred = cat([pred, current.expand(1, current.shape[0])])

                predictions = cat([predictions, pred])
                # backward pass
                mse = self.loss.compute_loss(pred, y_train[k*mini_batch_size : (k+1)*mini_batch_size])

                grad_wrt_output = self.loss.compute_grad(pred, y_train[k*mini_batch_size : (k+1)*mini_batch_size])
                for i in range(current_batch.shape[0]):
                    current_grad = grad_wrt_output[i]
                    for layer in reversed(self.layers):
                        current_grad = layer.backward(current_grad, step_size=0.001)

                errors.append(npmean(mse))

            print('Training MSE at epoch {} : {}'.format(epoch, npmean(errors)))

        return predictions


    def predict(self):
        pass

    def check_consistency(self, layer):
        if len(self.layers) == 0 and layer.get_input_size() == self.input_size:
            return True
        elif len(self.layers) > 0:
            if self.layers[-1].get_hidden_size() == layer.get_input_size():
                return True
        return False