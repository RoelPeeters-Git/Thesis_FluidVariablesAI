# -*- coding: utf-8 -*-
# %% ===== Required packages =====

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from GeneralFunctions import upgradedata
from tensorflow import keras as K
from tensorflow.keras.layers import Dense

# %% ===== Docstring =====
"""
Created on Sat Mar 7 2020

@author: Peeters Roel
"""
"""
Masterthesis project: Determining Fluid Variables with AI

This project will use the Tensorflow framework to train and utilize a
deep neural network.
The DNN utilizes simulation data from a CFD-model of airflow around a cilinder.
Input features are velocity-vectors in points in the wake of the cilinder.
The output desired is the force acting on the cilinder.
In first instance only the y-component features will be utilized.

The cylinder is being displaced in the airflow with multisine frequencies
at a relative amplitude. For each relative Amplitude a Model is trained
and evaluated. This Model uses the data from rel Amp 0,05
"""
# %% ===== Data import & preprocess =====

datapath = '../../DataFiles/Dataset_100Hz/relAmp0p05/'
X1 = pd.read_csv(datapath+'U_A0p05_100_1.csv', header=None,
                 index_col=0, dtype=np.float32)
X2 = pd.read_csv(datapath+'U_A0p05_100_2.csv', header=None,
                 index_col=0, dtype=np.float32)
X3 = pd.read_csv(datapath+'U_A0p05_100_3.csv', header=None,
                 index_col=0, dtype=np.float32)
X4 = pd.read_csv(datapath+'U_A0p05_100_4.csv', header=None,
                 index_col=0, dtype=np.float32)
X5 = pd.read_csv(datapath+'U_A0p05_100_5.csv', header=None,
                 index_col=0, dtype=np.float32)
Y1 = pd.read_csv(datapath+'force_A0p05_100_1.csv', header=None,
                 index_col=0, dtype=np.float32)
Y2 = pd.read_csv(datapath+'force_A0p05_100_2.csv', header=None,
                 index_col=0, dtype=np.float32)
Y3 = pd.read_csv(datapath+'force_A0p05_100_3.csv', header=None,
                 index_col=0, dtype=np.float32)
Y4 = pd.read_csv(datapath+'force_A0p05_100_4.csv', header=None,
                 index_col=0, dtype=np.float32)
Y5 = pd.read_csv(datapath+'force_A0p05_100_5.csv', header=None,
                 index_col=0, dtype=np.float32)

# Each Cy-value is represented by 3 timesteps of the Uy-measurements.
# This is simulated by adding data of timesteps t-1, t-2 to each timestep t.
# The first 2 timesteps are dropped.

Xtrain = [upgradedata(f) for f in [X1, X2, X3, X4]]
Xtrain = pd.concat(Xtrain)
Xtest = upgradedata(X5)

Ytrain = [f.iloc[2:, :] for f in [Y1, Y2, Y3, Y4]]
Ytrain = pd.concat(Ytrain)
Ytest = Y5.iloc[2:, :]

XTrain = Xtrain.to_numpy()
XDev = Xtest.iloc[0:len(Xtest)//2, :].to_numpy()
XTest = Xtest.iloc[len(Xtest)//2:-1, :].to_numpy()

YTrain = Ytrain.to_numpy()
YDev = Ytest.iloc[0:len(Ytest)//2, :].to_numpy()
YTest = Ytest.iloc[len(Ytest)//2:-1, :].to_numpy()

# %% ===== Build and train the model =====
# Training is done using the Xtrain, Ytrain datasets
# Validation is done using the Xdev, Ydev datasets
# The model consists of 6 dense layers with ELU activation functions
# and L2-regularization is implemented

regul = K.regularizers.l2(0.001)

inputs = K.Input(shape=(63,), name='Inputlayer')
x = Dense(63, activation='elu', name='Layer1',
          kernel_regularizer=regul)(inputs)
x = Dense(30, activation='elu', name='Layer2',
          kernel_regularizer=regul)(x)
x = Dense(15, activation='elu', name='Layer3',
          kernel_regularizer=regul)(x)
outputs = Dense(1, name='OutputLayer')(x)

Amp0p05_model = K.Model(inputs=inputs, outputs=outputs,
                        name='Model_relAmp0p05')

# Defining the parameters to train the DNN
optimizer = K.optimizers.Adam()
lossfunc = K.losses.MeanSquaredError()
metrics = [K.metrics.MeanSquaredError(), K.metrics.RootMeanSquaredError(),
           K.metrics.MeanAbsoluteError()]

EarlyStop = K.callbacks.EarlyStopping(monitor='val_loss', patience=50,
                                      verbose=1)

Amp0p05_model.compile(loss=lossfunc, optimizer=optimizer, metrics=metrics)

Amp0p05hist = Amp0p05_model.fit(XTrain, YTrain, epochs=500,
                                validation_data=(XDev, YDev),
                                callbacks=[EarlyStop])

hist = pd.DataFrame(Amp0p05hist.history)
K.utils.plot_model(Amp0p05_model, to_file='./Amp0p05_model.png',
                   show_shapes=True, show_layer_names=True)
hist.to_csv('Amp0p05_history.csv')
Amp0p05_model.save('Model_Amp0p05_100', save_format='tf')

# %% ===== Analyzing the model results =====

hist = pd.read_csv('Amp0p05_history.csv')
plt.figure(1)
plt.semilogy(hist.index, hist['mean_squared_error'], 'k', label='Train_MSE')
plt.semilogy(hist.index, hist['val_mean_squared_error'], 'b', label='Dev_MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('relAmp0p05_model')
plt.legend()
plt.savefig('Training_history.png')


# %% ===== Evaluate the model with the testset =====

Amp0p05_model = K.models.load_model('Model_Amp0p05_100/')
predics = Amp0p05_model.predict(XTest)
N = len(YTest)
error = (YTest-predics)
MSE = 1/N * np.sum(np.abs(error)**2)
relMSE = MSE / (1/N*np.sum(np.abs(YTest)**2))
print(f'Absolute error = {MSE}, Relative error = {relMSE}')

plt.figure(2)
plt.plot(error)
plt.xlabel('')
plt.ylabel('Error values')
plt.savefig('EvaluatedError.png')
