from utils.loading import load_data

from utils.data_augmentation import downsample

from conv_models import ConvNet3


if __name__ == '__main__':
    try:
        import torch
        assert torch.__version__[:5] == '0.4.0'

        print('We are training here our best model "ConvNet3" that can be found in conv_models.py.')
        print('Check the report of the project for more information.')
        print('\n\n')

        print('Loading the data.')
        print('\n\n')
        x_train, y_train = load_data(train=True, one_khz=False)
        x_train_one_khz, y_train_one_khz = load_data(train=True, one_khz=True)
        x_train_downsampled, y_train_downsampled = downsample(x_train_one_khz, y_train_one_khz)
        x_test, y_test = load_data(train=False, one_khz=False)

        lr_tuned = 0.000777
        noise_tuned = 1.1

        print('Training the model with the 100Hz training data.')
        model = ConvNet3()
        model.fit(x_train, y_train, epochs=50, lr=lr_tuned, batch_size=16,
                  standardize=True, noise=noise_tuned, crop=False, lr_decay=(10, 0.8))

        train_accuracy = model.compute_accuracy(x_train, y_train)
        test_accuracy = model.compute_accuracy(x_test, y_test)
        print('Accuracy on train set : {}%.'.format(round(100*train_accuracy, 4)))
        print('Accuracy on test set : {}%.'.format(round(100*test_accuracy, 4)))
    except AssertionError:
        print('Beware that the code was developed under torch 0.4.0.')
        print('You are currently running torch {}'.format(torch.__version__))