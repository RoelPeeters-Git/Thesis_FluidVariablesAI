# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 10:52:27 2020

Fully connected NN for MISO model of VIV CFD Data

Test batchnormalization and adding more layers
@author: Peeters Roel
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
import matplotlib.pyplot as plt
import seaborn as sns

# %% ===== Load datasets and create custom function =====

Utrain = pd.read_csv('Utrain.csv', header=None).to_numpy()
ftrain = pd.read_csv('ftrain.csv', header=None).to_numpy()
# Udev = pd.read_csv('Udev.csv', header=None).to_numpy()
# fdev = pd.read_csv('fdev.csv', header=None).to_numpy()
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


# %% ===== Define models =====
Model7 = K.Sequential([Dense(21, input_shape=[len(Utrain[1])], name='Hidden1'),
                       BatchNormalization(trainable=True),
                       Activation('relu', name='Act1'),
                       Dense(16, name='Hidden2'),
                       BatchNormalization(trainable=True),
                       Activation('relu', name='Act2'),
                       Dense(16, name='Hidden3'),
                       BatchNormalization(trainable=True),
                       Activation('relu', name='Act3'),
                       Dense(8, name='Hidden4'),
                       BatchNormalization(trainable=True),
                       Activation('relu', name='Act4'),
                       Dense(8, name='Hidden5'),
                       BatchNormalization(trainable=True),
                       Activation('relu', name='Act5'),
                       Dense(4, name='Hidden6'),
                       BatchNormalization(trainable=True),
                       Activation('relu', name='Act6'),
                       Dense(4,  name='Hidden7'),
                       BatchNormalization(trainable=True),
                       Activation('relu', name='Act7'),
                       Dense(1, activation='linear', name='Output')],
                      name='SWS_model7')
Model7.compile(loss=lossfunc, optimizer=optim, metrics=metric)

Model8 = K.Sequential([Dense(21, input_shape=[len(Utrain[1])], name='Hidden1'),
                       BatchNormalization(trainable=True),
                       Activation('relu', name='Act1'),
                       Dense(16, name='Hidden2'),
                       BatchNormalization(trainable=True),
                       Activation('relu', name='Act2'),
                       Dense(16, name='Hidden3'),
                       BatchNormalization(trainable=True),
                       Activation('relu', name='Act3'),
                       Dense(16, name='Hidden4'),
                       BatchNormalization(trainable=True),
                       Activation('relu', name='Act4'),
                       Dense(8, name='Hidden5'),
                       BatchNormalization(trainable=True),
                       Activation('relu', name='Act5'),
                       Dense(8, name='Hidden6'),
                       BatchNormalization(trainable=True),
                       Activation('relu', name='Act6'),
                       Dense(4, name='Hidden7'),
                       BatchNormalization(trainable=True),
                       Activation('relu', name='Act7'),
                       Dense(4, name='Hidden8'),
                       BatchNormalization(trainable=True),
                       Activation('relu', name='Act8'),
                       Dense(1, activation='linear', name='Output')],
                      name='SWS_model8')
Model8.compile(loss=lossfunc, optimizer=optim, metrics=metric)


Model9 = K.Sequential([Dense(21, input_shape=[len(Utrain[1])], name='Hidden1'),
                       BatchNormalization(trainable=True),
                       Activation('relu', name='Act1'),
                       Dense(16, name='Hidden2'),
                       BatchNormalization(trainable=True),
                       Activation('relu', name='Act2'),
                       Dense(16, name='Hidden3'),
                       BatchNormalization(trainable=True),
                       Activation('relu', name='Act3'),
                       Dense(16, name='Hidden4'),
                       BatchNormalization(trainable=True),
                       Activation('relu', name='Act4'),
                       Dense(8, name='Hidden5'),
                       BatchNormalization(trainable=True),
                       Activation('relu', name='Act5'),
                       Dense(8, name='Hidden6'),
                       BatchNormalization(trainable=True),
                       Activation('relu', name='Act6'),
                       Dense(8, name='Hidden7'),
                       BatchNormalization(trainable=True),
                       Activation('relu', name='Act7'),
                       Dense(4, name='Hidden8'),
                       BatchNormalization(trainable=True),
                       Activation('relu', name='Act8'),
                       Dense(4, name='Hidden9'),
                       BatchNormalization(trainable=True),
                       Activation('relu', name='Act9'),
                       Dense(1, activation='linear', name='Output')],
                      name='SWS_model9')
Model9.compile(loss=lossfunc, optimizer=optim, metrics=metric)


Model10 = K.Sequential([Dense(21, input_shape=[len(Utrain[1])],
                              name='Hidden1'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act1'),
                        Dense(21, name='Hidden2'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act2'),
                        Dense(16, name='Hidden3'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act3'),
                        Dense(16, name='Hidden4'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act4'),
                        Dense(16, name='Hidden5'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act5'),
                        Dense(8, name='Hidden6'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act6'),
                        Dense(8, name='Hidden7'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act7'),
                        Dense(8, name='Hidden8'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act8'),
                        Dense(4, name='Hidden9'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act9'),
                        Dense(4, name='Hidden10'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act10'),
                        Dense(1, activation='linear', name='Output')],
                       name='SWS_model10')
Model10.compile(loss=lossfunc, optimizer=optim, metrics=metric)


Model11 = K.Sequential([Dense(21, input_shape=[len(Utrain[1])],
                              name='Hidden1'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act1'),
                        Dense(21, name='Hidden2'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act2'),
                        Dense(21, name='Hidden3'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act3'),
                        Dense(16, name='Hidden4'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act4'),
                        Dense(16, name='Hidden5'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act5'),
                        Dense(16, name='Hidden6'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act6'),
                        Dense(8, name='Hidden7'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act7'),
                        Dense(8, name='Hidden8'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act8'),
                        Dense(8, name='Hidden9'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act9'),
                        Dense(4, name='Hidden10'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act10'),
                        Dense(4, name='Hidden11'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act11'),
                        Dense(1, activation='linear', name='Output')],
                       name='SWS_model10')
Model11.compile(loss=lossfunc, optimizer=optim, metrics=metric)


Model12 = K.Sequential([Dense(21, input_shape=[len(Utrain[1])],
                              name='Hidden1'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act1'),
                        Dense(21, name='Hidden2'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act2'),
                        Dense(21, name='Hidden3'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act3'),
                        Dense(16, name='Hidden4'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act4'),
                        Dense(16, name='Hidden5'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act5'),
                        Dense(16, name='Hidden6'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act6'),
                        Dense(16, name='Hidden7'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act7'),
                        Dense(8, name='Hidden8'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act8'),
                        Dense(8, name='Hidden9'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act9'),
                        Dense(8, name='Hidden10'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act10'),
                        Dense(4, name='Hidden11'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act11'),
                        Dense(4, name='Hidden12'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act12'),
                        Dense(1, activation='linear', name='Output')],
                       name='SWS_model10')
Model12.compile(loss=lossfunc, optimizer=optim, metrics=metric)

Model14 = K.Sequential([Dense(21, input_shape=[len(Utrain[1])],
                              name='Hidden1'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act1'),
                        Dense(21, name='Hidden2'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act2'),
                        Dense(21, name='Hidden3'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act3'),
                        Dense(16, name='Hidden4'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act4'),
                        Dense(16, name='Hidden5'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act5'),
                        Dense(16, name='Hidden6'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act6'),
                        Dense(16, name='Hidden7'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act7'),
                        Dense(8, name='Hidden8'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act8'),
                        Dense(8, name='Hidden9'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act9'),
                        Dense(8, name='Hidden10'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act10'),
                        Dense(8, name='Hidden11'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act11'),
                        Dense(4, name='Hidden12'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act12'),
                        Dense(4, name='Hidden13'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act13'),
                        Dense(4, name='Hidden14'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act14'),
                        Dense(1, activation='linear', name='Output')],
                       name='SWS_model10')
Model14.compile(loss=lossfunc, optimizer=optim, metrics=metric)

Model16 = K.Sequential([Dense(21, input_shape=[len(Utrain[1])],
                              name='Hidden1'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act1'),
                        Dense(21, name='Hidden2'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act2'),
                        Dense(21, name='Hidden3'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act3'),
                        Dense(21, name='Hidden4'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act4'),
                        Dense(16, name='Hidden5'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act5'),
                        Dense(16, name='Hidden6'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act6'),
                        Dense(16, name='Hidden7'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act7'),
                        Dense(16, name='Hidden8'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act8'),
                        Dense(8, name='Hidden9'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act9'),
                        Dense(8, name='Hidden10'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act10'),
                        Dense(8, name='Hidden11'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act11'),
                        Dense(8, name='Hidden12'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act12'),
                        Dense(4, name='Hidden13'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act13'),
                        Dense(4, name='Hidden14'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act14'),
                        Dense(4, name='Hidden15'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act15'),
                        Dense(4, name='Hidden16'),
                        BatchNormalization(trainable=True),
                        Activation('relu', name='Act16'),
                        Dense(1, activation='linear', name='Output')],
                       name='SWS_model10')
Model16.compile(loss=lossfunc, optimizer=optim, metrics=metric)

models = [7, 8, 9, 10, 11, 12, 14, 16]
nr_models = len(models)
relRMS_training = np.zeros((5, nr_models))
relRMS_validation = np.zeros((5, nr_models))

# %% ===== Training the models =====

for i in range(5):
    Model7hist = Model7.fit(Utrain, ftrain, epochs=tr_epoch,
                            batch_size=batchsize,
                            validation_data=(Utest, ftest),
                            validation_steps=5)
    Model8hist = Model8.fit(Utrain, ftrain, epochs=tr_epoch,
                            batch_size=batchsize,
                            validation_data=(Utest, ftest),
                            validation_steps=5)
    Model9hist = Model9.fit(Utrain, ftrain, epochs=tr_epoch,
                            batch_size=batchsize,
                            validation_data=(Utest, ftest),
                            validation_steps=5)
    Model10hist = Model10.fit(Utrain, ftrain, epochs=tr_epoch,
                              batch_size=batchsize,
                              validation_data=(Utest, ftest),
                              validation_steps=5)
    Model11hist = Model11.fit(Utrain, ftrain, epochs=tr_epoch,
                              batch_size=batchsize,
                              validation_data=(Utest, ftest),
                              validation_steps=5)
    Model12hist = Model12.fit(Utrain, ftrain, epochs=tr_epoch,
                              batch_size=batchsize,
                              validation_data=(Utest, ftest),
                              validation_steps=5)
    Model14hist = Model14.fit(Utrain, ftrain, epochs=tr_epoch,
                              batch_size=batchsize,
                              validation_data=(Utest, ftest),
                              validation_steps=5)
    Model16hist = Model16.fit(Utrain, ftrain, epochs=tr_epoch,
                              batch_size=batchsize,
                              validation_data=(Utest, ftest),
                              validation_steps=5)

    relRMS_training[i, 0] = Model7hist.history['rel_rmse'][-1]
    relRMS_training[i, 1] = Model8hist.history['rel_rmse'][-1]
    relRMS_training[i, 2] = Model9hist.history['rel_rmse'][-1]
    relRMS_training[i, 3] = Model10hist.history['rel_rmse'][-1]
    relRMS_training[i, 4] = Model11hist.history['rel_rmse'][-1]
    relRMS_training[i, 5] = Model12hist.history['rel_rmse'][-1]
    relRMS_training[i, 6] = Model14hist.history['rel_rmse'][-1]
    relRMS_training[i, 7] = Model16hist.history['rel_rmse'][-1]

    relRMS_validation[i, 0] = Model7hist.history['val_rel_rmse'][-1]
    relRMS_validation[i, 1] = Model8hist.history['val_rel_rmse'][-1]
    relRMS_validation[i, 2] = Model9hist.history['val_rel_rmse'][-1]
    relRMS_validation[i, 3] = Model10hist.history['val_rel_rmse'][-1]
    relRMS_validation[i, 4] = Model11hist.history['val_rel_rmse'][-1]
    relRMS_validation[i, 5] = Model12hist.history['val_rel_rmse'][-1]
    relRMS_validation[i, 6] = Model14hist.history['val_rel_rmse'][-1]
    relRMS_validation[i, 7] = Model16hist.history['val_rel_rmse'][-1]

# Run until here
# %%  ===== Plot the final loss =====

# rmstr = 'rel_rmse'
# rmsdev = 'val_rel_rmse'

# trainloss = [hist7[rmstr].iloc[-1], hist8[rmstr].iloc[-1],
#              hist9[rmstr].iloc[-1], hist10[rmstr].iloc[-1],
#              hist11[rmstr].iloc[-1], hist12[rmstr].iloc[-1]]
# devloss = [hist7[rmsdev].iloc[-1], hist8[rmsdev].iloc[-1],
#            hist9[rmsdev].iloc[-1], hist10[rmsdev].iloc[-1],
#            hist11[rmsdev].iloc[-1], hist12[rmsdev].iloc[-1]]

mean_relRMS_training = np.mean(relRMS_training, axis=0)
mean_relRMS_validation = np.mean(relRMS_validation, axis=0)

sns.set(context='paper', style='whitegrid')
fig, ax = plt.subplots()
plt.xticks(models, models)
ax.plot(models, mean_relRMS_training, 'b',
        label='Average rel_rmse on training')
ax.plot(models, mean_relRMS_validation, 'r',
        label='Average rel_rmse on validation')
ax.set(xlabel='Hidden layers in model',
       ylabel='Average rel_rmse')
ax.legend()
fig.suptitle('Average of 5 rel_RMSE of models with increasing layers using BatchNorm')
fig.show()

# %% ===== Save models and data =====

path_to_save = 'BatchNorm/'

models = [Model7, Model8, Model9, Model10, Model11, Model12]

Model7.save(path_to_save+'Model7', save_format='h5')
Model8.save(path_to_save+'Model8', save_format='h5')
Model9.save(path_to_save+'Model9', save_format='h5')
Model10.save(path_to_save+'Model10', save_format='h5')
Model11.save(path_to_save+'Model11', save_format='h5')
Model12.save(path_to_save+'Model12', save_format='h5')
Model14.save(path_to_save+'Model14', save_format='h5')
Model16.save(path_to_save+'Model16', save_format='h5')
