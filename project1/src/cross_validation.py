import numpy as np

from sklearn.model_selection import KFold


def compute_accuracy(y, y_hat):
    return np.sum(y == y_hat)/np.size(y)


def cross_validation(model_builder, x_train, y_train, num_folds=5, **kwargs):
    k_fold = KFold(n_splits=num_folds, shuffle=True)
    acc = 0
    for train_index, test_index in k_fold.split(x_train):
        model = model_builder()

        data = x_train[train_index]
        targets = y_train[train_index]

        validation_data = x_train[test_index]
        validation_targets = y_train[test_index]

        history = model.fit(data, targets, validation_data, validation_targets, **kwargs)
        arg_min = np.argmin(history['val_loss'])
        acc = acc + history['val_acc'][arg_min]
    return acc/num_folds
