from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from mlmodel import MLModel

from feature_augmentation_helper import augment_data


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
            else:
                raise ValueError('The argument "model" is invalid')
        elif isinstance(model, LogisticRegression) or isinstance(model, SVC) or isinstance(model, RandomForestClassifier):
            self.model = model
        else:
            raise ValueError('The argument "model is invalid"')
        return self.model

    def __init__(self, model='logistic', data=None, targets=None, feature_augmentation=None, **kwargs):
        MLModel.__init__(self)
        self.model = self.set_model(model)
        self.data = data
        self.targets = targets
        self.feature_augmentation = feature_augmentation
        self.set_params(**kwargs)

    def train(self, data=None, targets=None, feature_augmentation=None):
        if data is not None:
            self.data = data

        if targets is not None:
            self.targets = targets

        if feature_augmentation is not None:
            self.feature_augmentation = feature_augmentation

        self.data = augment_data(self.data, self.feature_augmentation)

        self.model.fit(self.data, self.targets)

    def predict(self, data):
        data = augment_data(data, self.feature_augmentation)

        return self.model.predict(data)