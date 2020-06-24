# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns

"""
Created on Thu May  7 18:25:03 2020

@author: Peeters Roel

Masterthesis project

Dense NN for MISO model of VIV CFD data
Models performance with increasing hidden layers compared
Each datapoint represents a timestep of 0.01 seconds.
"""

# %% ===== Load datasets and create custom function =====

Utrain = pd.read_csv('Utrain.csv', header=None).to_numpy()
ftrain = pd.read_csv('ftrain.csv', header=None).to_numpy()
Udev = pd.read_csv('Udev.csv', header=None).to_numpy()
fdev = pd.read_csv('fdev.csv', header=None).to_numpy()
Utest = pd.read_csv('Utest.csv', header=None).to_numpy()
ftest = pd.read_csv('ftest.csv', header=None).to_numpy()


def rel_rmse(y_true, y_pred):
    out = tf.math.sqrt(
        tf.math.reduce_mean(
            tf.math.square(y_pred-y_true))) / tf.math.sqrt(
                tf.math.reduce_mean(tf.math.square(y_true)))
    return out


# %% ===== Dense model parameters =====
features = (Utrain.shape[1],)
optim = K.optimizers.Adam()
lossfunc = K.losses.MeanSquaredError()
metric = [rel_rmse]
tr_epoch = 300
batchsize = 512
regul = K.regularizers.l2(0.001)
earlystop = K.callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                      verbose=1)

# kernel_regularizer=regul,
# %% ===== Different models with increasing layers to create =====
Model2 = K.Sequential([Dense(21, activation='relu',
                             input_shape=[len(Utrain[1])], name='Hidden1'),
                       Dense(16, activation='relu', name='Hidden2'),
                       Dense(1, activation='linear', name='Output')],
                      name='SWS_model2')
Model2.compile(loss=lossfunc, optimizer=optim, metrics=metric)
Model2hist = Model2.fit(Utrain, ftrain, epochs=tr_epoch, batch_size=batchsize,
                        validation_data=(Udev, fdev), validation_steps=5)
hist2 = pd.DataFrame(Model2hist.history)

Model3 = K.Sequential([Dense(21, activation='relu',
                             input_shape=[len(Utrain[1])], name='Hidden1'),
                       Dense(16, activation='relu', name='Hidden2'),
                       Dense(8, activation='relu', name='Hidden3'),
                       Dense(1, activation='linear', name='Output')],
                      name='SWS_model3')

Model3.compile(loss=lossfunc, optimizer=optim, metrics=metric)
Model3hist = Model3.fit(Utrain, ftrain, epochs=tr_epoch, batch_size=batchsize,
                        validation_data=(Udev, fdev), validation_steps=5)
hist3 = pd.DataFrame(Model3hist.history)

Model4 = K.Sequential([Dense(21, activation='relu',
                             input_shape=[len(Utrain[1])], name='Hidden1'),
                       Dense(16, activation='relu', name='Hidden2'),
                       Dense(8, activation='relu', name='Hidden3'),
                       Dense(8, activation='relu', name='Hidden4'),
                       Dense(1, activation='linear', name='Output')],
                      name='SWS_model4')
Model4.compile(loss=lossfunc, optimizer=optim, metrics=metric)
Model4hist = Model4.fit(Utrain, ftrain, epochs=tr_epoch, batch_size=batchsize,
                        validation_data=(Udev, fdev), validation_steps=5)
hist4 = pd.DataFrame(Model4hist.history)

Model5 = K.Sequential([Dense(21, activation='relu',
                             input_shape=[len(Utrain[1])], name='Hidden1'),
                       Dense(16, activation='relu', name='Hidden2'),
                       Dense(8, activation='relu', name='Hidden3'),
                       Dense(8, activation='relu', name='Hidden4'),
                       Dense(4, activation='relu', name='Hidden5'),
                       Dense(1, activation='linear', name='Output')],
                      name='SWS_model5')
Model5.compile(loss=lossfunc, optimizer=optim, metrics=metric)
Model5hist = Model5.fit(Utrain, ftrain, epochs=tr_epoch, batch_size=batchsize,
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
                      name='SWS_model6')
Model6.compile(loss=lossfunc, optimizer=optim, metrics=metric)
Model6hist = Model6.fit(Utrain, ftrain, epochs=tr_epoch, batch_size=batchsize,
                        validation_data=(Udev, fdev), validation_steps=5)
hist6 = pd.DataFrame(Model6hist.history)

Model7 = K.Sequential([Dense(21, activation='relu',
                             input_shape=[len(Utrain[1])], name='Hidden1'),
                       Dense(16, activation='relu', name='Hidden2'),
                       Dense(16, activation='relu', name='Hidden3'),
                       Dense(8, activation='relu', name='Hidden4'),
                       Dense(8, activation='relu', name='Hidden5'),
                       Dense(4, activation='relu', name='Hidden6'),
                       Dense(4, activation='relu', name='Hidden7'),
                       Dense(1, activation='linear', name='Output')],
                      name='SWS_model7')
Model7.compile(loss=lossfunc, optimizer=optim, metrics=metric)
Model7hist = Model7.fit(Utrain, ftrain, epochs=tr_epoch, batch_size=batchsize,
                        validation_data=(Udev, fdev), validation_steps=5)
hist7 = pd.DataFrame(Model7hist.history)

Model8 = K.Sequential([Dense(21, activation='relu',
                             input_shape=[len(Utrain[1])], name='Hidden1'),
                       Dense(16, activation='relu', name='Hidden2'),
                       Dense(16, activation='relu', name='Hidden3'),
                       Dense(16, activation='relu', name='Hidden4'),
                       Dense(8, activation='relu', name='Hidden5'),
                       Dense(8, activation='relu', name='Hidden6'),
                       Dense(4, activation='relu', name='Hidden7'),
                       Dense(4, activation='relu', name='Hidden8'),
                       Dense(1, activation='linear', name='Output')],
                      name='SWS_model8')
Model8.compile(loss=lossfunc, optimizer=optim, metrics=metric)
Model8hist = Model8.fit(Utrain, ftrain, epochs=tr_epoch, batch_size=batchsize,
                        validation_data=(Udev, fdev), validation_steps=5)
hist8 = pd.DataFrame(Model8hist.history)

Model9 = K.Sequential([Dense(21, activation='relu',
                             input_shape=[len(Utrain[1])], name='Hidden1'),
                       Dense(16, activation='relu', name='Hidden2'),
                       Dense(16, activation='relu', name='Hidden3'),
                       Dense(16, activation='relu', name='Hidden4'),
                       Dense(8, activation='relu', name='Hidden5'),
                       Dense(8, activation='relu', name='Hidden6'),
                       Dense(8, activation='relu', name='Hidden7'),
                       Dense(4, activation='relu', name='Hidden8'),
                       Dense(4, activation='relu', name='Hidden9'),
                       Dense(1, activation='linear', name='Output')],
                      name='SWS_model9')
Model9.compile(loss=lossfunc, optimizer=optim, metrics=metric)
Model9hist = Model9.fit(Utrain, ftrain, epochs=tr_epoch, batch_size=batchsize,
                        validation_data=(Udev, fdev), validation_steps=5)
hist9 = pd.DataFrame(Model9hist.history)

Model10 = K.Sequential([Dense(21, activation='relu',
                              input_shape=[len(Utrain[1])], name='Hidden1'),
                       Dense(21, activation='relu', name='Hidden2'),
                       Dense(16, activation='relu', name='Hidden3'),
                       Dense(16, activation='relu', name='Hidden4'),
                       Dense(16, activation='relu', name='Hidden5'),
                       Dense(8, activation='relu', name='Hidden6'),
                       Dense(8, activation='relu', name='Hidden7'),
                       Dense(8, activation='relu', name='Hidden8'),
                       Dense(4, activation='relu', name='Hidden9'),
                       Dense(4, activation='relu', name='Hidden10'),
                       Dense(1, activation='linear', name='Output')],
                       name='SWS_model10')
Model10.compile(loss=lossfunc, optimizer=optim, metrics=metric)
Model10hist = Model10.fit(Utrain, ftrain, epochs=tr_epoch,
                          batch_size=batchsize, validation_data=(Udev, fdev),
                          validation_steps=5)
hist10 = pd.DataFrame(Model10hist.history)


# %%  ===== Plot the final loss =====

rmstr = 'rel_rmse'
rmsdev = 'val_rel_rmse'
models = [2, 3, 4, 5, 6, 7, 8, 9, 10]
trainloss = [hist2[rmstr].iloc[-1], hist3[rmstr].iloc[-1],
             hist4[rmstr].iloc[-1], hist5[rmstr].iloc[-1],
             hist6[rmstr].iloc[-1], hist7[rmstr].iloc[-1],
             hist8[rmstr].iloc[-1], hist9[rmstr].iloc[-1],
             hist10[rmstr].iloc[-1]]
devloss = [hist2[rmsdev].iloc[-1], hist3[rmsdev].iloc[-1],
           hist4[rmsdev].iloc[-1], hist5[rmsdev].iloc[-1],
           hist6[rmsdev].iloc[-1], hist7[rmsdev].iloc[-1],
           hist8[rmsdev].iloc[-1], hist9[rmsdev].iloc[-1],
           hist10[rmsdev].iloc[-1]]


sns.set(context='paper', style='whitegrid')
fig, ax = plt.subplots()
plt.xticks(models, models)
ax.plot(models, trainloss, 'b', label='Rel_rmse on training')
ax.plot(models, devloss, 'r', label='Rel_rmse on validation')
ax.set(xlabel='Hidden layers in model', ylabel='Final rel_rmse')
ax.legend()
fig.suptitle('Training and validation loss of models with increasing layers')
fig.show()

# %% ===== Save models and data =====

path_to_save = 'Modelcompare/'

models = [Model2, Model3, Model4, Model5, Model6, Model7, Model8, Model9,
          Model10]

hist2.to_csv(path_to_save+'hist2.csv')
hist3.to_csv(path_to_save+'hist3.csv')
hist4.to_csv(path_to_save+'hist4.csv')
hist5.to_csv(path_to_save+'hist5.csv')
hist6.to_csv(path_to_save+'hist6.csv')
hist7.to_csv(path_to_save+'hist7.csv')
hist8.to_csv(path_to_save+'hist8.csv')
hist9.to_csv(path_to_save+'hist9.csv')
hist10.to_csv(path_to_save+'hist10.csv')

Model2.save(path_to_save+'Model2', save_format='h5')
Model3.save(path_to_save+'Model3', save_format='h5')
Model4.save(path_to_save+'Model4', save_format='h5')
Model5.save(path_to_save+'Model5', save_format='h5')
Model6.save(path_to_save+'Model6', save_format='h5')
Model7.save(path_to_save+'Model7', save_format='h5')
Model8.save(path_to_save+'Model8', save_format='h5')
Model9.save(path_to_save+'Model9', save_format='h5')
Model10.save(path_to_save+'Model10', save_format='h5')
