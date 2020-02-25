# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 14:21:45 2020

@author: Peeters Roel
"""
from tensorflow import keras as K
from tensorflow.keras.layers import Dense
import pandas as pd
# import matplotlib.pyplot as plt

# import the datasets from LoadDatasetSmall.py
from LoadDatasetSmall import Usingle_train, Csingle_train
from LoadDatasetSmall import Umulti_train, Cmulti_train
from LoadDatasetSmall import Usweep_train, Csweep_train

"""
Masterthesis project: Determining Fluid Variables with AI
Part 3

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

This script is only used to train different modellayouts and save the model to
a Tensorflow object.
"""

features = (63,)
Epochs = 1000
columnheaders = ['Loss', 'RMSE']


def build_4layer_model():
    """
    Builds a sequential, fully-connected TF-NN with 3 relu layers and
    1 linear layer.
    The optimizer, loss and metrics are predefined
    Arguments:
        None
    Output:
        model -- TF-NN
    """
    model = K.Sequential()
    model.add(Dense(32, input_shape=features, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))

    adam = K.optimizers.Adam()
    metric = K.metrics.RootMeanSquaredError()
    model.compile(loss='mse', optimizer=adam, metrics=[metric])

    return model


def build_8layer_model():
    """
    Builds a sequential, fully-connected TF-NN with 7 relu layers and
    1 linear layer.
    The optimizer, loss and metrics are predefined
    Arguments:
        None
    Output:
        model -- TF-NN
    """
    model = K.Sequential()
    model.add(Dense(32, input_shape=features, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))

    adam = K.optimizers.Adam()
    metric = K.metrics.RootMeanSquaredError()
    model.compile(loss='mse', optimizer=adam, metrics=[metric])

    return model


# %%   ======== Model for single sine wave dataset ==========
modelsingle = build_4layer_model()
histsingle = modelsingle.fit(Usingle_train, Csingle_train, epochs=Epochs,
                             batch_size=64, verbose=0)
histsingle = pd.DataFrame(histsingle.history)
histsingle.columns = columnheaders
histsingle.to_csv('HistObject_SingleSine.csv')

modelsingle.evaluate(Usingle_train, Csingle_train, batch_size=32,
                     verbose=1)
modelsingle.save('modelsingle', save_format='tf')


# %%   ======= 4layer model for multisine wave dataset ========

modelmulti1 = build_4layer_model()
histmulti1 = modelmulti1.fit(Umulti_train, Cmulti_train, epochs=Epochs,
                             batch_size=64, verbose=0)
histmulti1 = pd.DataFrame(histmulti1.history)
histmulti1.columns = columnheaders
histmulti1.to_csv('HistObject_MultiSine1.csv')

modelmulti1.save('modelmulti1', save_format='tf')

# %% ========== 8layer model for multisine wave dataset =========

modelmulti2 = build_8layer_model()
histmulti2 = modelmulti2.fit(Umulti_train, Cmulti_train, epochs=Epochs,
                             batch_size=64, verbose=0)
histmulti2 = pd.DataFrame(histmulti2.history)
histmulti2.columns = columnheaders
histmulti2.to_csv('HistObject_MultiSine1.csv')

modelmulti2.save('modelmulti2', save_format='tf')

# %% =========== 4layer model for sweepsine wave dataset ==========

modelsweep1 = build_4layer_model()
histsweep1 = modelsweep1.fit(Usweep_train, Csweep_train, epochs=Epochs,
                             batch_size=64, verbose=0)
histsweep1 = pd.DataFrame(histsweep1.history)
histsweep1.columns = columnheaders
histsweep1.to_csv('HistObject_SweepSine1.csv')

modelsweep1.save('modelsweep1', save_format='tf')

# %% ===========8layer model for sweepsine wave dataset ==========

modelsweep2 = build_8layer_model()
histsweep2 = modelsweep2.fit(Usweep_train, Csweep_train, epochs=Epochs,
                             batch_size=64, verbose=0)
histsweep2 = pd.DataFrame(histsweep2.history)
histsweep2.columns = columnheaders
histsweep2.to_csv('HistObject_SweepSine2.csv')

modelsweep2.save('modelsweep2', save_format='tf')
