# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from NN_utils_v3_1 import L_layer_model

"""
Created on Wed May  6 16:56:02 2020

@author: Peeters Roel

Neural Network as Universal approximator
This python script will use the numpy functions to build a NN for a random
non-linear function.
It will be demonstrated that a NN with a single hidden layer will approximate
this function within a given error. The activation function chosen is the
sigmoid.
"""


# %% ===== Building the continuous function and a training and test set =====
def cont_func(t):
    y = 0.8*t**3
    return y


plt.clf()
plt.close('all')
sns.set(style='ticks')

t = np.linspace(0, 1, 500)  # x-training set
t.shape = (1, 500)
f_t = cont_func(t)          # y-training set
# xtest = np.linspace(0.2, 0.6, 50)
# xtest.shape = (1, 50)
# ytest = cont_func(xtest)

fig_data, ax_data = plt.subplots()
trainplot = ax_data.plot(t[0], f_t[0], label='Training data')
# testplot = ax_data.plot(xtest[0], ytest[0], '*', label='Test data')
ax_data.set_xlabel('Input')
ax_data.set_ylabel('Output')

# %% ===== Training a NN with 1 hidden layer =====

inputshape = f_t.shape[0]
layers_dim = (inputshape, 10, 5, 1)
alpha = 0.01
num_iter = 2000
UAmodel = L_layer_model(t, f_t, layers_dim, 'relu', learning_rate=alpha,
                        num_iterations=num_iter, print_cost=False,
                        optimizer='GD')
parameters, gradients, costs, predictions = UAmodel

train_error = np.abs(predictions - f_t)
avg_train_error = np.mean(train_error)
rel_train_error = np.sqrt(np.mean(train_error**2)/np.mean(f_t**2))
final_cost = costs[-1]

predplot = ax_data.plot(t[0], predictions[0], 'g--', label='Predictions')
ax_data.legend(loc='upper left')
fig_data.show()

fig_cost, ax_cost = plt.subplots()
ax_cost.plot(np.squeeze(costs))
ax_cost.set_yscale('log')
ax_cost.set_ylabel('Cost')
ax_cost.set_xlabel('Iteration (x10)')
ax_cost.set_title(f'Learning rate = {alpha}')
fig_cost.show()

# %% ===== Collect all required data to store =====

Training_data = pd.Series(f_t[0], index=t[0])
Pred_data = pd.Series(predictions[0], index=t[0])
Function_data = pd.DataFrame([Training_data, Pred_data])
Function_data.to_csv('./NNUA_function6.csv')

data = {'Modelled Function': ['f(t) = 0.8*t^3'],
        'Final cost': [final_cost],
        'Average Training Error': [avg_train_error],
        'Relative Training Error': [rel_train_error],
        }
NN_layout = {'Dimensions': layers_dim, 'Learning rate': [alpha],
             'Activation function': ['ReLU', 'linear'],
             'Training epochs': [num_iter],
             'Optimizer': ['Gradient Descent']
             }

pd.DataFrame.from_dict(data, orient='index').to_csv('./NNUA_Data6.csv')
pd.DataFrame.from_dict(NN_layout, orient='index').to_csv('./NNUA_layout6.csv')

# NN_layers = pd.DataFrame([pd.Series(parameters['W1'].reshape(100,)),
#                           pd.Series(parameters['b1'].reshape(100,)),
#                           pd.Series(parameters['W2'].reshape(100,)),
#                           pd.Series(parameters['b2'].reshape(1,))],
#                          index=['W1', 'b1', 'W2', 'b2']).T
# NN_layers.to_csv('./NNUA_parameters6.csv')
