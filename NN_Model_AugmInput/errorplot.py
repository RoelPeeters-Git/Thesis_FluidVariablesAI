# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
# import tensorflow as tf
from tensorflow import keras as K
# from tensorflow.keras.layers import Dense
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

# %% ===== Evaluate on test data and calculate some metrics =====

AugmModel1 = K.models.load_model('AugmModel1')
# SFModel1.summary()
AugmModel1_eval = AugmModel1.evaluate(Utest, ftest)
f_pred = AugmModel1.predict(Utest)
assert(f_pred.shape == ftest.shape)
error = ftest - f_pred

time = np.arange(0, len(ftest)*0.01, step=0.01)
time.shape = (len(time), 1)
assert(time.shape == ftest.shape)

# ===== plot of the predictions and error =====
sns.set(context='paper', style='whitegrid')

# fig, axs = plt.subplots(2, 1, sharex=True)
# axs[0].plot(time[750:2750], ftest[750:2750], 'b--', label='cy true values')
# axs[1].plot(time[750:2750], f_pred[750:2750], 'r:',
#             label='cy predicted values')
# axs[1].plot(time[750:2750], error[750:2750], 'k', label='Error')
# axs[0].set(ylabel='C_y True value')
# axs[1].set(xlabel='Time (s)', ylabel='C_y predicted / Error')

# fig.suptitle('Predictions and error of the trained model', fontsize=14)
# fig.legend(loc='upper right')

plt.figure(1)
plt.subplot(211)
plt.plot(time, ftest, 'b--', label='cy true values')
plt.ylabel('C_y True value', fontsize=24)
plt.subplot(212)
plt.plot(time, f_pred, 'r--', label='cy predicted values')
plt.plot(time, error, 'k', label='Error')
plt.xlabel('Time (s)', fontsize=24)
plt.ylabel('C_y predicted / Error', fontsize=24)
plt.suptitle('Predictions and error of the trained model', fontsize=32)
