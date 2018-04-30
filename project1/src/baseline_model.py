import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from feature_augmentation_helper import augment_features

from mlmodel import MLModel


class BaselineModel(MLModel):

    def set_params(self, **kwargs):
        self.model.set_params(**kwargs)

    def set_model(self, model):
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
        elif isinstance(model, LogisticRegression) or isinstance(model, SVC) or isinstance(model, RandomForestClassifier):
            self.model = model
        else:
            raise ValueError('The argument "model" is invalid')
        return self.model

    def update_data(self, data=None, targets=None, feature_augmentation=None, scale=True):
        MLModel.update_data(self, data, targets, feature_augmentation)

        if data is not None:
            self.data = self.data.view(data.shape[0], -1)
            self.data = self.data.numpy().astype(float)
            if scale:
                self.scaler.fit(self.data)
                # self.data = self.scaler.transform(self.data)

    def __init__(self, model='logistic', data=None, targets=None, feature_augmentation=None, **kwargs):
        MLModel.__init__(self)
        self.model = self.set_model(model)
        self.scaler = preprocessing.StandardScaler()
        self.update_data(data=data, targets=targets, feature_augmentation=feature_augmentation)
        self.set_params(**kwargs)

    def fit(self, data=None, targets=None, feature_augmentation=None):
        self.update_data(data=data, targets=targets, feature_augmentation=feature_augmentation)
        self.model.fit(self.data, self.targets)

    def predict(self, data):
        data = augment_features(data, self.feature_augmentation)
        data = data.view(data.shape[0], -1)
        data = data.numpy().astype(float)
        data = self.scaler.transform(data)

        return self.model.predict(data)

    def cross_validation(self, data=None, targets=None, feature_augmentation=None, num_folds=5, scoring=None):
        self.update_data(data=data, targets=targets, feature_augmentation=feature_augmentation, scale=False)
        scalar = preprocessing.StandardScaler()
        pipeline = Pipeline([('transformer', scalar), ('estimator', self.model)])
        scores = cross_val_score(pipeline, self.data, self.targets, cv=num_folds, scoring=scoring)
        return np.mean(scores), np.std(scores)

    def tune_params(self, params, data=None, targets=None, feature_augmentation=None, num_folds=5, scoring=None):
        self.update_data(data=data, targets=targets, feature_augmentation=feature_augmentation, scale=False)
        scalar = preprocessing.StandardScaler()
        pipeline = Pipeline([('transformer', scalar),
                             ('gridsearch', GridSearchCV(self.model, params, cv=num_folds,
                                                         scoring=scoring, return_train_score=True))])
        # return pipeline.fit(self.data, self.targets)
        return pipeline.score
