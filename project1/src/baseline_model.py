"""
File defining the class BaselineModel inheriting the class MLModel. It is used to create and train baseline models
such as logistic regressions, support vector machines and random forests.
"""


import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from utils.feature_augmentation import augment_features

from mlmodel import MLModel


class BaselineModel(MLModel):

    def __init__(self, model='logistic', data=None, targets=None, feature_augmentation=None,
                 standardize=False, normalize=False, **kwargs):
        """
        Initializer.
        :param model: String defining the model type that is used. Can be 'logistic', 'svm' or 'forest'.
        :param data: Raw data (transformations are made automatically).
        :param targets: Targets of the data.
        :param feature_augmentation: Feature augmentation function applied on the data.
        :param standardize: If True the data will be standardized.
        :param normalize: If True the data will be scaled in the range [0, 1]
        :param kwargs: Additional parameters that can be given to the specific sklearn model.
        """
        MLModel.__init__(self)
        self.model = self.set_model(model)
        self.normalizer = None
        self.update_data(data=data, targets=targets, feature_augmentation=feature_augmentation,
                         standardize=standardize, normalize=normalize)
        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        """
        Set the parameters of the sklearn model.
        :param kwargs: Parameters.
        :return: Nothing.
        """
        self.model.set_params(**kwargs)

    def set_model(self, model):
        """
        Set the sklearn model.
        :param model: Model name. Can be 'logistic', 'svm', or 'forest'.
        :return: The sklearn model that was built.
        """
        if isinstance(model, str):
            model = model.lower()
            if model == 'logistic':
                self.model = LogisticRegression()
            elif model == 'svm' or model == 'svc':
                self.model = SVC()
            elif model == 'rfc' or model == 'forest':
                self.model = RandomForestClassifier()
            elif model == 'rfc' or model == 'forest':
                self.model = RandomForestClassifier()
            elif model == 'markov' or model == 'hmm':
                raise NotImplementedError()
                # To be completed
            else:
                raise ValueError('The argument "model" is invalid')
        elif isinstance(model, LogisticRegression) or isinstance(model, SVC) or isinstance(model,
                                                                                           RandomForestClassifier):
            self.model = model
        else:
            raise ValueError('The argument "model" is invalid')
        return self.model

    def update_data(self, data=None, targets=None, feature_augmentation=None, standardize=False, normalize=False):
        """
        Update the data used to train the model.
        :param data: Raw data (transformations are made automatically).
        :param targets: Labels of the data.
        :param feature_augmentation: Feature augmentation function applied on the data.
        :param standardize: If True the data will be standardized.
        :param normalize: If True the data will be scaled in the range [0, 1]
        :return: Nothing.
        """
        MLModel.update_data(self, data, targets, feature_augmentation)

        if data is not None:
            self.data = self.data.view(data.shape[0], -1)
            self.data = self.data.numpy().astype(float)
            if standardize:
                self.normalizer = preprocessing.StandardScaler()
                self.normalizer.fit(self.data)
            elif normalize:
                self.normalizer = preprocessing.MinMaxScaler()
                self.normalizer.fit(self.data)
            else:
                self.normalizer = None

            if self.normalizer is not None:
                self.data = self.normalizer.transform(self.data)

    def fit(self, data=None, targets=None, **kwargs):
        """
        Method used to fit the model to some data and targets.
        :param data: Raw data (transformations are made automatically).
        :param targets: Labels of the data.
        :param kwargs: Additional parameters that can be given to the method update_data.
        :return: Nothing.
        """
        self.update_data(data=data, targets=targets, **kwargs)
        self.model.fit(self.data, self.targets)

    def predict(self, data):
        """
        Method used to predict the label of new data.
        :param data: Raw data.
        :return: Predicted labels.
        """
        data = augment_features(data, self.feature_augmentation)
        data = data.view(data.shape[0], -1)
        data = data.numpy().astype(float)
        if self.normalizer is not None:
            data = self.normalizer.transform(data)

        return self.model.predict(data)

    def cross_validation(self, data=None, targets=None, num_folds=5, scoring=None, raw=False, **kwargs):
        """
        Performs cross validation to evaluate the model performances by splitting the data into several folds.
        :param data: Raw data.
        :param targets: Labels of the data.
        :param num_folds: Number of folds for the cross validation.
        :param scoring: Type of scoring used.
        :param raw: If True, the list of the scores are returned (instead of the mean and std).
        :param kwargs: Additional parameters that can be given to the update_data method.
        :return: A tuple (mean, std) with the mean of the scores and their standard deviation.
        """
        self.update_data(data=data, targets=targets, **kwargs)
        scores = cross_val_score(self.model, self.data, self.targets, cv=num_folds, scoring=scoring)
        if raw:
            return scores
        else:
            return np.mean(scores), np.std(scores)

    def tune_params(self, params, data=None, targets=None, num_folds=5, scoring=None, **kwargs):
        """
        Performs a grid search cross validation to tune hyper-parameters.
        :param params: Dictionary containing the hyper-parameters to be applied.
        :param data: Raw data.
        :param targets: Labels of the data.
        :param num_folds: Number of folds for the cross validation.
        :param scoring: Type of scoring used.
        :param kwargs: Additional parameters that can be given to the update_data method.
        :return: A gridSearchCV object which fitted the data and contains the scores.
        """
        self.update_data(data=data, targets=targets, **kwargs)
        clf = GridSearchCV(self.model, params, cv=num_folds, scoring=scoring, return_train_score=True)
        clf.fit(self.data, self.targets)
        return clf

    def compute_accuracy(self, x_test, y_test):
        """
        Compute the accuracy of the model for some new data and new labels given.
        :param x_test: Raw data.
        :param y_test: Labels.
        :return: The accuracy of the model.
        """
        y_hat = self.predict(x_test)
        y_test = y_test.numpy()
        return np.sum(y_test == y_hat)/np.size(y_test)
