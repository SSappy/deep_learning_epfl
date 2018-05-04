# Project 1

The objective of this project is to train a predictor of finger movements from Electroencephalography
(EEG) recordings.

[Subject](https://github.com/SSappy/deep_learning_epfl/blob/master/project1/doc/miniproject-1.pdf)

Team members :
- Armand Boschin
- Quentin Rebjock
- Shengzhao Lei

### Usage

### Requirements

### Organization of the folder
* data : contains the data used for the project.
    * pickle : folder containing the pickle files that were saved in order not to recompute them.
    * labels_data_set_iv.txt : labels for the test data set.
    * sp1s_aa_test.txt : 100Hz frequency testing data set.
    * sp1s_aa_test_1000Hz.txt : 1000Hz frequency testing data set.
    * sp1s_aa_train.txt : 100Hz frequency training data set and labels.
    * sp1s_aa_train_1000Hz.txt : 1000Hz frequency training data set and labels.
* doc : documentation of the project.
    * miniproject-1.pdf : subject
* src : source code of the project.
    * mlmodel.py : base class for baseline and neural network models.
    * baseline_model.py : class defining a baseline model (logistic, svm, random forest, or hidden markov).
    * nnmodel.py : class defining a neural network model.
    * loading.py : contains functions to load the data set and save objects.
    * cross_validation.py : contains functions to do cross validation to select models.
    * data_augmentation.py : contains classes defining data augmentation methods.
    * feature_augmentation.py : contains functions to augment features in a 2d data set.
    * bci_dataset.py : class containing the data set for nn models, helping the training with handy methods.
    * dlc_bci.py : helper functions to load the data set (the higher level functions in loading.py should be used instead).
    * conv_models.py : contains the convolutional neural network models used in this project.