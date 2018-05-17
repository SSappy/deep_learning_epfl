import sys

from distutils import util

from utils.loading import load_data


def build_best_model():
    x_train, y_train = load_data(train=True, one_khz=False)
    from conv_models import ConvNet1Dropout
    best_model = ConvNet1Dropout()
    lr_tuned = 0.000777
    noise_tuned = 1.1
    best_model.fit(x_train, y_train, epochs=30, lr=lr_tuned, batch_size=4,
                   standardize=True, noise=noise_tuned, crop=False, lr_decay=(10, 0.8))
    return best_model
    raise NotImplementedError()


def load_best_model():
    raise NotImplementedError()


if __name__ == '__main__':
    try:
        import torch
        assert torch.__version__[:5] == '0.4.0'

        train = False
        if len(sys.argv) > 1:
            train = bool(util.strtobool(sys.argv[1]))
        if train:
            model = build_best_model()
        else:
            model = load_best_model()
        x_test, y_test = load_data(train=False, one_khz=False)
        accuracy = model.compute_accuracy(x_test, y_test)
        print('The accuracy of the model on the testing set is {}%.'.format(round(100*accuracy, 4)))

    except AssertionError:
        print('Beware that the code was developed under torch 0.4.0.')
        print('You are currently running torch {}'.format(torch.__version__))