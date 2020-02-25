# -*- coding: utf-8 -*-
# %% ======== importing required packages =========

import pandas as pd
# import numpy as np
import tensorflow as tf
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


# %% ======= Loading required datasets =======
Xdev = pd.read_csv('../DatasetLarge/UyMS_DevData.csv', header=None,
                   index_col=False).to_numpy(dtype='float32')
Ydev = pd.read_csv('../DatasetLarge/CyMS_DevData.csv', header=None,
                   index_col=False).to_numpy(dtype='float32')
Xtest = pd.read_csv('../DatasetLarge/UyMS_TestData.csv', header=None,
                    index_col=False).to_numpy(dtype='float32')
Ytest = pd.read_csv('../DatasetLarge/CyMS_TestData.csv', header=None,
                    index_col=False).to_numpy(dtype='float32')

# %% ======= Loading the model =========
FluidModel = K.models.load_model('./FluidModel')

# %% ======= Evaluate model for dev and testsets ======
FluidModel.evaluate(Xdev, Ydev, batch_size=64, verbose=1)
loss = FluidModel.evaluate(Xtest, Ytest, batch_size=256)
