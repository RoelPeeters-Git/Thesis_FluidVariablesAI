# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:05:57 2019

@author: Peeters Roel
"""

""" 
Masterthesis project: Determining fluid variables with AI

TRAINING A NEURAL NETWORK TO CALCULATE FORCE COEFFICIENT 

This file will train a neural network with a set of given simulation data.
The design of the neural network can be changed from shallow to deep neural network.
Its parameters will be trained and evaluated versus a test set. 

The force on a cilinder in a flow of air is determined by the velocity field of the air.
In this model, a 1D-coëfficiënt of the force is calculated given 1D-component of the velocity field.

The simulation data consists of:
    - input: y-component of velocity field measured by 33 probes in wake of cilinder
    - output: Cy-component of the force acting on the cilinder
    
Different layouts of the DNN may be tested, to find the best NN architecture to tackle this problem.
"""

##### loading required packages and functions #####

import matplotlib.pyplot as plt
import pandas as pd
from load_data_JD import Xtrain, Xtest, Ytrain, Ytest
from NN_utils_v3 import predict_test, L_layer_model, gradient_check

##### Initial hyperparameters to start training a NN #####
plt.close('all')
xi = Xtrain.shape[0]
learning_rate = 0.02
layers_dims = [xi, 33, 33, 16, 5, 1]
num_layers = len(layers_dims)
num_iterations = 20000
activation = 'linear'

#%%
##### Training the NN with the given training sets #####

parameters, grads, costs = L_layer_model(Xtrain, Ytrain, layers_dims, activation, learning_rate, num_iterations, True, 'Adam')

#%%
##### Checking the trained NN with the test data #####

traintest, trainerror, trainacc = predict_test(Xtrain, Ytrain, parameters, activation)
devtest, deverror, devacc = predict_test(Xtest, Ytest, parameters, activation)

print('Accuracy on training set is ' + str(trainacc))
print('Accuracy on test set is ' + str(devacc))

##### performing a gradient check on a parameter #####

difference = gradient_check(Xtrain, Ytrain, parameters, 'W2', (0,6), grads, activation)
print('Gradient check difference is ' + str(difference))

fig, ax = plt.subplots(2)
fig.suptitle('Layers: '+str(layers_dims) + ' + Final layer: '+str(activation) + ' + Learning rate: '+str(learning_rate)) 
ax[0].plot(Ytest.T, 'r')
ax[0].plot(devtest.T, 'b')
ax[0].set_title('Ytest vs devtest')
ax[1].plot(deverror.T, 'k')
ax[1].set_title('Error of devtest vs true Ytest')
#%%  
##### Storing data in a pandas dataframe #####

df1 = pd.DataFrame({'Time of test': pd.Timestamp.now(), 'NN Layers': (layers_dims), 'Learning rate': learning_rate, 
                   '# of iterations': num_iterations, 'Final activation': activation, 'Final cost': costs[-1], 
                   'Training Set accuracy': trainacc, 'Dev Set accuracy': devacc, 'Gradient Check': difference})
#df2 = pd.DataFrame(parameters)

df1.to_csv('Data_1DCilinder.csv', mode = 'a')
#df2.to_csv('Parameters_1DCilinder.csv', mode = 'a')