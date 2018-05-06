import pickle

from utils import dlc_bci as bci


def load_data(train=True, one_khz=False):
    """
    Loads the training and testing data set. For example:
    x_train, y_train = load_data()
    :param train: If True, the training data set is returned, otherwise it's the testing one
    :param one_khz: If True, the data set sampled with 1kHz frequency is returned, otherwise the one with 100Hz
    :return: The corresponding data set
    """
    return bci.load(root='../data', train=train, one_khz=one_khz)


def save_obj(obj, name):
    """
    Saves an object in a pickle file.
    :param obj: Object to save.
    :param name: Name of the file.
    :return: Nothing.
    """
    with open('../data/pickle/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """
    Loads an object from a pickle file.
    :param name: File name.
    :return: Loaded object.
    """
    with open('../data/pickle/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
