# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 15:59:36 2019

@author: Peeters Roel
"""
from tensorflow import keras as K
from tensorflow.keras import layers
from tensorflow import saved_model
from LoadData_1D import load_data
import matplotlib.pyplot as plt
import pandas as pd
# import numpy as np
"""
Masterthesis project: Determining Fluid Variables with AI
Part 2

This project will use the Tensorflow framework to train and utilize a
deep neural network.
The DNN utilizes simulation data from a CFD-model of airflow around a cilinder.
Input features are velocity-vectors in points in the wake of the cilinder.
The output desired is the force acting on the cilinder.
In first instance only the y-component features will be utilized.
"""


def build_sequential_model(first_layer, layer, optimizer, loss, metrics):
    """
    Builds a sequential, fully-connected TF-NN
    Arguments:
        first_layer -- the first layer's features
                        (units, inputshape, activation)
        layer -- subsequent layers' features (units, activation)
        optimizer -- the optimizer to use in the model
        loss -- the loss function of the model
        metrics -- the metrics to compare the predictions vs the true values
    Output:
        model -- TF-NN
    """
    model = K.Sequential()
    model.add(layers.Dense(units=first_layer[0],
                           input_shape=[first_layer[1]],
                           activation=first_layer[2]))

    for unit, activation in layer:
        model.add(layers.Dense(units=unit, activation=activation))

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[metrics])
    return model

# model = K.Sequential([
#            layers.Dense(21, input_shape = [inputshape], activation ='relu'),
#            layers.Dense(10, activation = 'relu'),
#            layers.Dense(1, activation = 'linear')
#            ])


next_layers = [(20, 'relu'), (10, 'relu'), (1, 'linear')]
Optimizer = K.optimizers.SGD()
metric = K.metrics.RootMeanSquaredError()
lossfunc = 'mse'
EPOCHS = 100
# %%
# ======== Building Model around first dataset ===========

Xtrain1, Xtest1, Ytrain1, Ytest1 = load_data(1)
inputshape1 = Xtrain1.shape[1]

first_layer1 = [inputshape1, inputshape1, 'relu']

model1 = build_sequential_model(first_layer1, next_layers,
                                Optimizer, lossfunc, metric)
model1.summary()

history1 = model1.fit(Xtrain1, Ytrain1, epochs=EPOCHS)

hist1 = pd.DataFrame(history1.history)
hist1['epoch'] = history1.epoch

saved_model.save(model1, 'FirstSavedModel')
hist1.to_csv('Model1_Results.csv')

loss1, rmse1 = model1.evaluate(Xtest1, Ytest1, verbose=0)

y_hat1 = model1.predict(Xtest1)
error1 = Ytest1-y_hat1

fig1, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(Ytest1, '--b', label='True Values')
axes[0].plot(y_hat1, ':r', label='Predicted Values')
axes[0].legend()
axes[1].plot(error1, '-k', label='Error')
fig1.suptitle('Simple Sine-wave velocity field, dataset 1')
fig1.show()

# %%
# ======== Building Model around 2nd dataSet =========
Xtrain2, Xtest2, Ytrain2, Ytest2 = load_data(2)
inputshape2 = Xtrain2.shape[1]

first_layer2 = [inputshape2, inputshape2, 'relu']
# # next_layers = [(inputshape2, 'relu'), (inputshape2, 'relu'),
#                (inputshape2, 'relu'), (15, 'relu'),
#                (15, 'relu'), (1, 'linear')]


model2 = build_sequential_model(first_layer2, next_layers,
                                Optimizer, lossfunc, metric)

history2 = model2.fit(Xtrain2, Ytrain2, epochs=EPOCHS)
hist2 = pd.DataFrame(history2.history)

saved_model.save(model2, 'SecondSavedModel')
hist2.to_csv('Model2_Results2.csv')
loss2, rmse2 = model2.evaluate(Xtest2, Ytest2, verbose=0)
y_hat2 = model2.predict(Xtest2)
error2 = Ytest2 - y_hat2

fig2, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(Ytest2, '--b', label='True Value')
axes[0].plot(y_hat2, ':r', label='Predicted Value')
axes[0].legend()
axes[1].plot(error2, '-k', label='Error signal')
axes[1].legend()
fig2.suptitle('Multiple Sine-wave velocity field, dataset 2')
fig2.show()

# %%
# ========== Building Model around third dataset  =============
Xtrain3, Xtest3, Ytrain3, Ytest3 = load_data(3)
inputshape3 = Xtrain3.shape[1]

first_layer3 = [inputshape3, inputshape3, 'relu']

model3 = build_sequential_model(first_layer3, next_layers,
                                Optimizer, lossfunc, metric)

history3 = model3.fit(Xtrain3, Ytrain3, epochs=EPOCHS)
hist3 = pd.DataFrame(history3.history)

saved_model.save(model3, 'ThirdSavedModel')
hist3.to_csv('Model2_Results3.csv')
loss3, rmse3 = model3.evaluate(Xtest3, Ytest3, verbose=0)
y_hat3 = model3.predict(Xtest3)
error3 = Ytest3 - y_hat3

fig3, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(Ytest3, '--b', label='True Value')
axes[0].plot(y_hat3, ':r', label='Predicted Value')
axes[0].legend()
axes[1].plot(error3, '-k', label='Error signal')
axes[1].legend()
fig3.suptitle('Multiple Sine-wave velocity field, dataset 3')
fig3.show()
