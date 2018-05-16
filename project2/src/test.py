from sequential import Sequential
from linear import Linear
from activations import ReLU, Tanh
from losses import MSE
from utils import build_data
from time import time

def build_model():
    model = Sequential(MSE(), input_size=2)
    model.add_layer(Linear(2, 25))
    model.add_layer(ReLU(25))
    model.add_layer(Linear(25, 25))
    model.add_layer(ReLU(25))
    model.add_layer(Linear(25, 25))
    model.add_layer(Tanh(25))
    model.add_layer(Linear(25, 2))
    return model


def build_and_train_model(n=1000, epochs=70, step_size=0.01):
    print('\nGenerating training, validation and test sets of size {} each.'.format(n))
    x_train, y_train = build_data(n)
    x_validation, y_validation = build_data(n)
    x_test, y_test = build_data(n)
    print('Building the model\n')
    model = build_model()
    model.summary()
    print('\nTraining the model on {} epochs'.format(epochs))
    t = time()
    model.fit(x_train, y_train, x_validation, y_validation, epochs=epochs, step_size=step_size)
    t = int(time()- t)
    print('\nTraining time : {0:0.0f} seconds ({1:0.3f} seconds per epoch)'.format(t, t/epochs))
    print('\nTesting the trained model :')
    te_predictions, te_loss = model.predict(x_test, y_test)
    te_accuracy = (te_predictions == y_test).sum() / y_test.shape[1] / te_predictions.shape[0]
    print('Test loss : {}'.format(te_loss.mean()))
    print('Test accuracy : {}'.format(te_accuracy))

if __name__ == '__main__':
    try:
        import torch
        assert torch.__version__[:5] == '0.3.1'
        import warnings

        warnings.filterwarnings("ignore",
                                message="other is not broadcastable to self, but they have the same number of elements.  "
                                        "Falling back to deprecated pointwise behavior.")
        build_and_train_model()


    except AssertionError:
        print('Beware that the code was developed under torch 0.3.1.')
        print('You are currently running torch {}'.format(torch.__version__))
        if torch.__version__ == '0.4.0':
            print('You should downgrade to O.3.1 before using.')
        else:
            print('Make sure to upgrade to 0.3.1 before using.')
    