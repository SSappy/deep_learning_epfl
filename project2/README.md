# Project 2

The objective of this project is to design a mini deep learning framework using only pytorch's tensor operations and the standard math library, hence in particular without using autograd or the neural-network modules.

[Subject](https://github.com/SSappy/deep_learning_epfl/blob/master/project2/doc/miniproject-2.pdf)

Team members : 

* Armand Boschin
* Quentin Rebjock
* Shengzhao Lei

### Usage:
For a demonstration of the mini framework please run 
`$ python3 run.py` in a python3 environment with torch 0.3.1 installed.

This python script builds a model with three hidden fully connected layers (25 hidden units each) and ReLU, ReLU, Tanh activation layers. 

This model is trained to classify points in [0, 1]^2 depending on their distance to the central point (see `report.pdf` for more details).

### Requirements: 
The mini framework was developed and tested under Python 3.5.3 on Debian Stretch using Torch 0.3.1. It was tested on torch 0.4.0 but there are some issues. If you are running torch 0.4.0, we recommend that you downgrade before using.
### Organization of the folder:
* doc : documentation of the project including 
	* report 
	* subject

* src : source code of the mini framework
	* module.py : contains the interface from which other modules inherit.
	* test.py : file to be run for demonstration of the mini framework
	* activations.py : contains the activation layer modules (ReLU and Tanh).
	* linear.py : contains the linear layer module (fully connected hidden layer)/
	* losses.py : contains the loss module MSE.
	* sequential.py : contains the sequential model (several layers and training and prediction methods).
	* utils.py : contains functions to generate training and validation data.

