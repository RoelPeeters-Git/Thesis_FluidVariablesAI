# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 19:29:28 2019

@author: peete
"""

"""
Masterthesis project: Determining fluid variables with AI
version: 1.0

The network will be trained using randomly generated input data. 
The output values have a relation to the input according to the Frobenius-norm, i.e. $y = sum(x_i^2).
This model will have an output consisting of 1 real scalar.

The functions used in this algorithm are defined in the NN_utils_v1_0 module.

The code and functions are heavily inspired by the Deep learning specialization course on Coursera by Andrew Ng.
"""

import numpy as np
import matplotlib.pyplot as plt

from NN_utils_v1_1 import L_layer_model, predict_test, predict

# defining parameters for data generation

x_i = 5               # number of x per training example
m = 10                 # number of training examples
k = 5                 # number of dev examples
n = 5                 # number of test examples

#%% generate the data for training, development and final testing

x_train = np.around(np.random.rand(x_i,m), decimals = 2)
y_train = np.sum(x_train**2, axis=0, keepdims=True)

#print(x_train)
#print(x_train.shape)
#print(y_train)
#print(y_train.shape)

#%%

# defining hyperparameters for use in the NN, making it easy to tune the NN

L = 3                 # number of layers in NN
learning_rate = 0.001        # the learning rate /alpha 
layers_dims = [x_train.shape[0], m, 1]        # number of hidden units in each layer
num_iterations = 1000

assert(len(layers_dims) == L)

parameters = L_layer_model(x_train,y_train, layers_dims, learning_rate, num_iterations, True)

print(parameters)