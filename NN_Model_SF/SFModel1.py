# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
# import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns

"""
Created on Thu May  7 18:25:03 2020

@author: Peeters Roel

Masterthesis project

A dense NN is created and trained to model the wake velocity of the cylinder
to the force coefficient acting on it.
The datasets created are loaded, the model created and trained and then tested
for performance.
This model uses the ReLU-function in the hidden layers.
Each datapoint represents a timestep of 0.01 seconds.
"""

# %% ===== Load datasets and create tensorflow datasets =====

Utrain = pd.read_csv('Utrain.csv', header=None).to_numpy()
ftrain = pd.read_csv('ftrain.csv', header=None).to_numpy()
Udev = pd.read_csv('Udev.csv', header=None).to_numpy()
fdev = pd.read_csv('fdev.csv', header=None).to_numpy()
Utest = pd.read_csv('Utest.csv', header=None).to_numpy()
ftest = pd.read_csv('ftest.csv', header=None).to_numpy()


# %% ===== Create Dense model =====

features = (Utrain.shape[1],)
optim = K.optimizers.Adam()
lossfunc = K.losses.MeanSquaredError()
metric = [K.metrics.RootMeanSquaredError(), K.metrics.MeanAbsoluteError()]
regul = K.regularizers.l2(0.001)

SFModel1 = K.Sequential([Dense(21, activation='relu',
                               input_shape=[len(Utrain[1])],
                               kernel_regularizer=regul, name='Hidden1'),
                        Dense(16, activation='relu', kernel_regularizer=regul,
                              name='Hidden2'),
                        Dense(16, activation='relu', kernel_regularizer=regul,
                              name='Hidden3'),
                        Dense(8, activation='relu', kernel_regularizer=regul,
                              name='Hidden4'),
                        Dense(8, activation='relu', kernel_regularizer=regul,
                              name='Hidden5'),
                        Dense(4, activation='relu', kernel_regularizer=regul,
                              name='Hidden6'),
                        Dense(4, activation='relu', kernel_regularizer=regul,
                              name='Hidden7'),
                        Dense(1, activation='linear', name='Output')],
                        name='SFModel1')

SFModel1.compile(loss=lossfunc, optimizer=optim, metrics=metric)

earlystop = K.callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                      verbose=1)

SFModel1_hist = SFModel1.fit(Utrain, ftrain, epochs=1000, batch_size=128,
                             callbacks=[earlystop],
                             validation_data=(Udev, fdev), validation_steps=5)

hist = pd.DataFrame(SFModel1_hist.history)
hist.to_csv('SFModel1_hist.csv')

# plot model for later use
K.utils.plot_model(SFModel1, to_file='./SFModel1.png', show_shapes=True,
                   show_layer_names='True')

# save model
SFModel1.save('SFModel1', save_format='h5')


# %% ===== Evaluate on test data and calculate some metrics =====

SFModel1 = K.models.load_model('SFModel1')
# SFModel1.summary()
SFModel1_eval = SFModel1.evaluate(Utest, ftest)
f_pred = SFModel1.predict(Utest)
assert(f_pred.shape == ftest.shape)

time = np.arange(0, len(ftest)*0.01, step=0.01)
time.shape = (len(time), 1)
assert(time.shape == ftest.shape)

# Metrics for performance of trained network
num_trainsamples = len(ftrain)
num_testsamples = len(ftest)
num_params = SFModel1.count_params()
error = ftest - f_pred

abs_error = np.abs(error)
avg_error = np.mean(error)
std_error = np.std(error)
loss = np.sum(error**2) / num_testsamples
rel_error = loss / (np.sum((ftest)**2)/num_testsamples)
rel_RMSE = np.sqrt(rel_error)

SFModel1_perfo = {'Model activations': 'ReLU-Linear',
                  'Number of training samples': num_trainsamples,
                  'Number of model parameters': num_params,
                  'Ratio training samples-parameters':
                  num_trainsamples/num_params,
                  'Model loss on test data': SFModel1_eval[0],
                  'Model RMSE on test data': SFModel1_eval[1],
                  'Model Absolute error on test data': SFModel1_eval[2],
                  'Number of test samples': num_testsamples,
                  'Average Error': avg_error,
                  'Standard Deviation of Error': std_error,
                  'Average Absolute Error': np.mean(abs_error),
                  'Maximum Error': np.max(abs_error),
                  'Loss': loss,
                  'Relative Error': rel_error,
                  'Relative RMSE': rel_RMSE}

Model1Perfo = pd.DataFrame.from_dict(SFModel1_perfo, orient='index')
Model1Perfo.index.name = 'Model1'
Model1Perfo.to_csv('Model1Perfo.csv', header=['Value'])
Model1Perfo.to_latex('Model1Perfo.tex', header=['Value'])
# ===== plot of the predictions and error =====
sns.set(context='paper', style='whitegrid')

fig, axs = plt.subplots(2,1,sharex=True)
axs[1].plot(time[750:2750], ftest[750:2750], 'b--', label='cy true values')
axs[2].plot(time[750:2750], f_pred[750:2750], 'r:', label='cy predicted values')
axs[2].plot(time[750:2750], error[750:2750], 'k', label='Error')
axs.set(xlabel='Time (s)', ylabel='C_y/Error')

fig.suptitle('Predictions and error of the trained model', fontsize=14)
fig.legend(loc='upper right')
