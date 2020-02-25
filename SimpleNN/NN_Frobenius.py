# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 19:29:28 2019

@author: peete
"""

"""
Masterthesis project: Determining fluid variables with AI

The network will be trained using randomly generated input data. 
The Numpy Rand-function will be used, so all input data will be generated as a uniform distribution between 0 and 1.
The output values have a relation to the input according to the Frobenius-norm, i.e. $y = sum(x_i^2).
This model will have an output consisting of 1 real scalar.

In this instance a Neural Network is generated with fixed hyperparameters and standard gradient descent is used.

Underlying functions that train the model and makes new predictions are stored in NN_utils_V1_1.py

The code and functions are heavily inspired by the Deep learning specialization course on Coursera by Andrew Ng.
"""

import numpy as np
import matplotlib.pyplot as plt

from NN_utils_v2 import L_layer_model, predict_test, gradient_check

#%% defining parameters for data generation

x_i = 20               # number of x per training example
m = 1                # number of training examples
k = 10                 # number of dev examples


#%% generate the data for training, development and final testing

x_train = np.around(np.random.rand(x_i,m), decimals = 3)
y_train = np.sum(x_train**2, axis=0, keepdims=True)

x_dev = np.around(np.random.rand(x_i,k), decimals = 3)
y_dev = np.sum(x_dev**2, axis=0, keepdims=True)


#%% 
##### training the model to obtain parameters  #####

# defining hyperparameters for use in the NN, making it easy to tune the NN

L = 2                # number of layers in NN
learning_rate = 0.1       # the learning rate /alpha 
layers_dims = [x_train.shape[0], 1]        # number of hidden units in each layer
num_iterations = 10000
activation = 'linear'

assert(len(layers_dims) == L)

parameters, gradients, costs = L_layer_model(x_train, y_train, layers_dims, activation, learning_rate, num_iterations, True, optimizer = 'GD')

#%% 
##### Testing the trained neural network #####

# Both the training and test set will be fed into the model
# This will enable us to check the avoidable bias and variance of the model

training_test, training_error, training_acc = predict_test(x_train, y_train, parameters, activation)
dev_test, dev_error, dev_acc = predict_test(x_dev,y_dev, parameters, activation)

print('Final cost = ' + str(costs[-1]))
print('Training set accuracy = ' + str(training_acc))
print('Test set accuracy = ' + str(dev_acc))

#%%
##### gradient checking of the trained neural network #####

grad_diff_W2 = gradient_check(x_train, y_train, parameters, 'W1', (0,2), gradients, activation)

print('Gradient check for W1 is ' + str(grad_diff_W2))

plt.figure()
#plt.plot(dev_test.T, 'r')
plt.plot(y_dev.T, 'b')
plt.plot(dev_error.T, 'k')