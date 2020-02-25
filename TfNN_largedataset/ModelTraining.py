# -*- coding: utf-8 -*-
# %% ======== importing required packages =========

import pandas as pd
# import numpy as np
# import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.layers import Dense


# %% ======= Docstring =============
"""
Created on Mon Feb  3 20:36:18 2020

@author: Peeters Roel
"""

"""
Masterthesis project: Determining Fluid Variables with AI
Part 4

This project will use the Tensorflow framework to train and utilize a
deep neural network.
The DNN utilizes simulation data from a CFD-model of airflow around a cilinder.
Input features are velocity-vectors in points in the wake of the cilinder.
The output desired is the force acting on the cilinder.
In first instance only the y-component features will be utilized.

The input features have been modified, so that each Cy(n)-value is represented
with Uy(n)-values of Uy(n), Uy(n-1) and Uy(n-2).
This simulates the dependency of Cy(n) to the past values of the velocityfield.
The NN will be verified with a true Tf-RNN in order to compare which method
provides better results

Here a large dataset is being used to train a network around a range of
frequencies and amplitudes of the flow patterns around a cilinder.
The base frequency is the Stroahl-frequency of 3Hz and the amplitude is
normalized to the diametre of the cilinder.
"""


# %%  ======== importing the training datasets ========

Xtrain = pd.read_csv('C://Users/peete/VUB/Masterthesis/Code/DatasetLarge/UyMS_TrainData.csv',
                     header=None, index_col=False).to_numpy(dtype='float32')
Ytrain = pd.read_csv('C://Users/peete/VUB/Masterthesis/Code/DatasetLarge/CyMS_TrainData.csv',
                     header=None, index_col=False).to_numpy(dtype='float32')
Xdev = pd.read_csv('C://Users/peete/VUB/Masterthesis/Code/DatasetLarge/UyMS_DevData.csv',
                   header=None, index_col=False).to_numpy(dtype='float32')
Ydev = pd.read_csv('C://Users/peete/VUB/Masterthesis/Code/DatasetLarge/CyMS_DevData.csv',
                   header=None, index_col=False).to_numpy(dtype='float32')

# %%  ======== function to create a model with predetermined architecture ===


def build_model(features):
    """
    Builds a sequential, fully-connected TF-NN with 6 relu layers and
    1 linear layer.
    The optimizer, loss and metrics are predefined
    Arguments:
        features -- number of input features per training example
    Output:
        model -- TF-NN
    """

    model = K.Sequential()
    model.add(Dense(63, input_shape=features, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))

    optimizer = K.optimizers.Adam()
    loss = 'mse'
    metrics = [K.metrics.RootMeanSquaredError(), 'mae']

    model.compile(loss=loss, optimizer=optimizer, metric=metrics)

    return model


# %% ========= Setting parameters for training the NN =======
features = (Xtrain.shape[1],)
Epochs = 1000

EarlyStop = K.callbacks.EarlyStopping(monitor='loss', patience=50, verbose=1)
Board = K.callbacks.TensorBoard(log_dir='./board', write_graph=True,
                                profile_batch=0)

# %% ========= building and training the Model ==========
FluidModel = build_model(features)

history = FluidModel.fit(Xtrain, Ytrain,
                         batch_size=64, epochs=Epochs, verbose=0,
                         validation_data=(Xdev, Ydev),
                         callbacks=[EarlyStop, Board])

# %% ========== processing the model for visualization =========
K.utils.plot_model(FluidModel, to_file='./FluidModel.png',
                   show_shapes=True, show_layer_names=True)

FluidModel.save('FluidModel', save_format='tf')
hist = pd.DataFrame(history.history)

