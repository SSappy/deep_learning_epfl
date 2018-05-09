import numpy as np

from sklearn.model_selection import KFold


def compute_accuracy(y, y_hat):
    """
    Compute the accuracy of predicted labels given true labels.
    :param y: True labels.
    :param y_hat: Predicted labels
    :return: Accuracy.
    """
    return np.sum(y == y_hat)/np.size(y)


def early_stopping(history):
    """
    Artificially performs early stopping.
    :param history: Training history returned by the fit function.
    :return: The loss, validation loss, accuracy and validation accuracy at the early stopping point.
    """
    arg_min = np.argmin(history['val_loss'])
    return history['loss'][arg_min], history['val_loss'][arg_min], history['acc'][arg_min], history['val_acc'][arg_min]


def cross_validation(model_builder, x_train, y_train, num_folds=5, raw=False, history=False, **kwargs):
    """
    Applies cross validation to evaluate a model's performances.
    :param model_builder: Function returning a built model or model class for which the accuracy is evaluated.
    :param x_train: Raw data.
    :param y_train: Labels of the data.
    :param num_folds: Number of folds to split the data.
    :param raw: If True, returns all the accuracies instead of their mean and std.
    :param history: If True, returns all the histories instead of the accuracies.
    :param kwargs: Additional arguments given to the fit method of the model.
    :return: Mean of accuracies and standard deviation or list of accuracies or histories (depending on the parameters).
    """
    k_fold = KFold(n_splits=num_folds, shuffle=True)
    acc = []
    for train_index, test_index in k_fold.split(x_train):
        model = model_builder()

        data = x_train[train_index]
        targets = y_train[train_index]

        validation_data = x_train[test_index]
        validation_targets = y_train[test_index]

        history_ = model.fit(data, targets, validation_data, validation_targets, **kwargs)
        if history:
            acc.append(history_)
        else:
            arg_min = np.argmin(history_['val_loss'])
            acc.append(history_['val_acc'][arg_min])
    if raw:
        return acc
    else:
        return np.mean(acc), np.std(acc)
