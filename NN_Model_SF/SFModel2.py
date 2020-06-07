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
This model uses the ELU-function in the hidden layers.
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

SFModel2 = K.Sequential([Dense(21, activation='elu',
                               input_shape=[len(Utrain[1])],
                               kernel_regularizer=regul, name='Hidden1'),
                        Dense(16, activation='elu', kernel_regularizer=regul,
                              name='Hidden2'),
                        Dense(16, activation='elu', kernel_regularizer=regul,
                              name='Hidden3'),
                        Dense(8, activation='elu', kernel_regularizer=regul,
                              name='Hidden4'),
                        Dense(8, activation='elu', kernel_regularizer=regul,
                              name='Hidden5'),
                        Dense(4, activation='elu', kernel_regularizer=regul,
                              name='Hidden6'),
                        Dense(4, activation='elu', kernel_regularizer=regul,
                              name='Hidden7'),
                        Dense(1, activation='linear', name='Output')],
                        name='SFModel2')

SFModel2.compile(loss=lossfunc, optimizer=optim, metrics=metric)

earlystop = K.callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                      verbose=1)

SFModel2_hist = SFModel2.fit(Utrain, ftrain, epochs=1000, batch_size=128,
                             callbacks=[earlystop],
                             validation_data=(Udev, fdev), validation_steps=5)

hist = pd.DataFrame(SFModel2_hist.history)
hist.to_csv('SFModel2_hist.csv')

# plot model for later use
K.utils.plot_model(SFModel2, to_file='./SFModel2.png', show_shapes=True,
                   show_layer_names='True')

# save model
SFModel2.save('SFModel2', save_format='h5')


# %% ===== Evaluate on test data and calculate some metrics =====

SFModel2.summary()
SFModel2_eval = SFModel2.evaluate(Utest, ftest)
f_pred = SFModel2.predict(Utest)
assert(f_pred.shape == ftest.shape)

time = np.arange(0, len(ftest)*0.01, step=0.01)
time.shape = (len(time), 1)
assert(time.shape == ftest.shape)

# Metrics for performance of trained network
num_trainsamples = len(ftrain)
num_testsamples = len(ftest)
num_params = SFModel2.count_params()
error = (f_pred - ftest)

abs_error = np.abs(error)
avg_error = np.mean(error)
std_error = np.std(error)
loss = np.sum(error**2) / num_testsamples
rel_error = loss / (np.sum(np.abs(ftest)**2)/num_testsamples)
rel_RMSE = np.sqrt(rel_error)

SFModel2_perfo = {'Model activations': 'ELU-Linear',
                  'Ratio training samples-parameters':
                  num_trainsamples/num_params,
                  'Model loss on test data': SFModel2_eval[0],
                  'Model RMSE on test data': SFModel2_eval[1],
                  'Model Absolute error on test data': SFModel2_eval[2],
                  'Number of test samples': num_testsamples,
                  'Average Error': avg_error,
                  'Standard Deviation of Error': std_error,
                  'Average Absolute Error': np.mean(abs_error),
                  'Maximum Error': np.max(abs_error),
                  'Loss': loss,
                  'Relative Error': rel_error,
                  'Relative RMSE': rel_RMSE}

Model2Perfo = pd.DataFrame.from_dict(SFModel2_perfo, orient='index')
Model2Perfo.index.name = 'Model2'
Model2Perfo.to_csv('Model2Perfo.csv', header=['Value'])
# ===== plot of the predictions and error =====
sns.set(context='paper', style='whitegrid')
fig, ax = plt.subplots()
ax.plot(time, ftest, 'b--', label='cy test values')
ax.plot(time, f_pred, 'r:', label='cy predicted values')
ax.plot(time, error, 'k', label='Error')
ax.set(xlabel='Time (s)', ylabel='C_y/Error')


fig.suptitle('Output of Trained Model2', fontsize=14)
fig.legend(loc='upper right')
