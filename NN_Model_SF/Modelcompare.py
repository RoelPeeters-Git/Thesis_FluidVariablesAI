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

hist7 = pd.read_csv('SFModel1_hist.csv', index_col=None, header=[0])


# %% ===== Dense model parameters =====

features = (Utrain.shape[1],)
optim = K.optimizers.Adam()
lossfunc = K.losses.MeanSquaredError()
metric = [K.metrics.RootMeanSquaredError(), K.metrics.MeanAbsoluteError()]
regul = K.regularizers.l2(0.001)
earlystop = K.callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                      verbose=1)

# kernel_regularizer=regul,
# %% ===== Different models with increasing layers to create =====
Model2 = K.Sequential([Dense(21, activation='relu',
                             input_shape=[len(Utrain[1])], name='Hidden1'),
                       Dense(16, activation='relu', name='Hidden2'),
                       Dense(1, activation='linear', name='Output')],
                      name='SFModel1')
Model2.compile(loss=lossfunc, optimizer=optim, metrics=metric)
Model2hist = Model2.fit(Utrain, ftrain, epochs=150, batch_size=128,
                        validation_data=(Udev, fdev), validation_steps=5)
hist2 = pd.DataFrame(Model2hist.history)

Model3 = K.Sequential([Dense(21, activation='relu',
                             input_shape=[len(Utrain[1])], name='Hidden1'),
                       Dense(16, activation='relu', name='Hidden2'),
                       Dense(8, activation='relu', name='Hidden3'),
                       Dense(1, activation='linear', name='Output')],
                      name='SFModel1')

Model3.compile(loss=lossfunc, optimizer=optim, metrics=metric)
Model3hist = Model3.fit(Utrain, ftrain, epochs=150, batch_size=128,
                        validation_data=(Udev, fdev), validation_steps=5)
hist3 = pd.DataFrame(Model3hist.history)

Model4 = K.Sequential([Dense(21, activation='relu',
                             input_shape=[len(Utrain[1])], name='Hidden1'),
                       Dense(16, activation='relu', name='Hidden2'),
                       Dense(8, activation='relu', name='Hidden3'),
                       Dense(8, activation='relu', name='Hidden4'),
                       Dense(1, activation='linear', name='Output')],
                      name='SFModel1')
Model4.compile(loss=lossfunc, optimizer=optim, metrics=metric)
Model4hist = Model4.fit(Utrain, ftrain, epochs=150, batch_size=128,
                        validation_data=(Udev, fdev), validation_steps=5)
hist4 = pd.DataFrame(Model4hist.history)

Model5 = K.Sequential([Dense(21, activation='relu',
                             input_shape=[len(Utrain[1])], name='Hidden1'),
                       Dense(16, activation='relu', name='Hidden2'),
                       Dense(8, activation='relu', name='Hidden3'),
                       Dense(8, activation='relu', name='Hidden4'),
                       Dense(4, activation='relu', name='Hidden5'),
                       Dense(1, activation='linear', name='Output')],
                      name='SFModel1')
Model5.compile(loss=lossfunc, optimizer=optim, metrics=metric)
Model5hist = Model5.fit(Utrain, ftrain, epochs=150, batch_size=128,
                        validation_data=(Udev, fdev), validation_steps=5)
hist5 = pd.DataFrame(Model5hist.history)

Model6 = K.Sequential([Dense(21, activation='relu',
                             input_shape=[len(Utrain[1])], name='Hidden1'),
                       Dense(16, activation='relu', name='Hidden2'),
                       Dense(16, activation='relu', name='Hidden3'),
                       Dense(8, activation='relu', name='Hidden4'),
                       Dense(8, activation='relu', name='Hidden5'),
                       Dense(4, activation='relu', name='Hidden6'),
                       Dense(1, activation='linear', name='Output')],
                      name='SFModel1')
Model6.compile(loss=lossfunc, optimizer=optim, metrics=metric)
Model6hist = Model6.fit(Utrain, ftrain, epochs=150, batch_size=128,
                        validation_data=(Udev, fdev), validation_steps=5)
hist6 = pd.DataFrame(Model6hist.history)

# %%  ===== Plot the final loss =====

rmstr = 'root_mean_squared_error'
rmsdev = 'val_root_mean_squared_error'
models = [2, 3, 4, 5, 6]
trainloss = [hist2[rmstr].iloc[-1], hist3[rmstr].iloc[-1],
             hist4[rmstr].iloc[-1], hist5[rmstr].iloc[-1],
             hist6[rmstr].iloc[-1]]
devloss = [hist2[rmsdev].iloc[-1], hist3[rmsdev].iloc[-1],
           hist4[rmsdev].iloc[-1], hist5[rmsdev].iloc[-1],
           hist6[rmsdev].iloc[-1]]

sns.set(context='paper', style='whitegrid')
fig, ax = plt.subplots()
plt.xticks(models, models)
ax.plot(models, trainloss, 'b', label='Training RMSE')
ax.plot(models, devloss, 'r', label='Validation RMSE')
ax.set(xlabel='Hidden layers in model', ylabel='Final RMSE')
ax.legend()
fig.suptitle('Training and validation loss of models with increasing layers')
fig.show()

# %% ===== Evaluation of Model2 =====

pred2 = Model2.predict(Utest)
error2 = pred2 - ftest
relrmse = np.sqrt(np.mean(error2**2)/np.mean(ftest**2))
