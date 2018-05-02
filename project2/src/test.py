from sequential import Sequential
from linear import Linear
from activations import ReLU, Tanh
from losses import MSE
from utils import build_data

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


def build_and_train_model(n=1000, epochs=50, step_size=0.01):
    print('\nGenerating training and validation sets of size {} each.'.format(n))
    x_train, y_train = build_data(n)
    x_validation, y_validation = build_data(n)
    print('Building the model\n')
    model = build_model()
    model.summary()
    print('\nTraining the model on {} epochs'.format(epochs))
    model.fit(x_train, y_train, x_validation, y_validation, epochs=epochs, step_size=step_size)

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
    