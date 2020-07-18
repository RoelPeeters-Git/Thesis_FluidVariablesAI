# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 12:47:34 2020

Fully connected NN for MIMO model of VIV CFD Data:
    Map of Uy as input to cy and y as output

@author: Peeters Roel
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns


def rel_rmse(y_true, y_pred):
    out = tf.math.sqrt(
        tf.math.reduce_mean(
            tf.math.square(y_pred-y_true))) / tf.math.sqrt(
                tf.math.reduce_mean(tf.math.square(y_true)))
    return out


# %% ===== Import datasets =====

Utrainset = pd.read_csv('Utrainset.csv', header=None).to_numpy()
fytrainset = pd.read_csv('fytrainset.csv', header=None).to_numpy()
Utestset = pd.read_csv('Utestset.csv', header=None).to_numpy()
fytestset = pd.read_csv('fytestset.csv', header=None).to_numpy()


# %% ===== Set the hyperparameters =====

path_to_save = '/MIMO_disp/'
features = len(Utrainset[1])
outputs = len(fytrainset[1])
optim = K.optimizers.Adam()
lossfunc = K.losses.MeanSquaredError()
metric = [rel_rmse]
tr_epoch = 2000
batchsize = 655
# board = K.callbacks.TensorBoard(log_dir='./MIMO_disp/Tensorboard')

# regul = K.regularizers.l2(0.001)
# earlystop = K.callbacks.EarlyStopping(monitor='val_loss', patience=20,
#                                       verbose=1)

# %% ===== Function to build models =====


def build_model(layers, features, optimizer, loss, metric):
    """
    Parameters
    ----------
    layers : tuple
        number of neurons in layer, len(layers)= number of layers
    features : tuple
        number of input features

    Returns
    -------
    model = keras model.

    """
    model = K.Sequential(name=f'SeqModel{len(layers)}')
    model.add(K.Input(shape=features))
    count = 1

    for layer in layers[:-1]:
        model.add(Dense(layer, activation='relu',
                        name=f'Hidden{count}'))
        count += 1

    model.add(Dense(layers[-1], activation='linear', name='Output'))
    # model.summary()

    model.compile(optimizer=optimizer, loss=loss, metrics=metric)

    return model


# %% ===== Define model layers dimensions =====

layer12 = [16, 16, 16, 12, 12, 12, 8, 8, 8, 4, 4, outputs]
layer13 = [16, 16, 16, 16, 12, 12, 12, 8, 8, 8, 4, 4, outputs]
layer14 = [16, 16, 16, 16, 12, 12, 12, 12, 8, 8, 8, 4, 4, outputs]
layer15 = [16, 16, 16, 16, 16, 12, 12, 12, 12, 8, 8, 8, 4, 4, outputs]
layer16 = [16, 16, 16, 16, 16, 12, 12, 12, 12, 12, 8, 8, 8, 4, 4, outputs]

# %% ===== Train the neural networks =====
epoch = np.arange(1, tr_epoch+1)

model12_final = np.zeros((5, 4), dtype='float32')
model13_final = np.zeros((5, 4), dtype='float32')
model14_final = np.zeros((5, 4), dtype='float32')
model15_final = np.zeros((5, 4), dtype='float32')
model16_final = np.zeros((5, 4), dtype='float32')


for i in range(1, 6):

    model12 = build_model(layer12, features, optim, lossfunc, metric)
    hist12 = model12.fit(x=Utrainset, y=fytrainset, batch_size=batchsize,
                         epochs=tr_epoch, verbose=0,
                         validation_data=(Utestset, fytestset))
    hist12 = pd.DataFrame(hist12.history, index=epoch)
    hist12.to_csv(f'MIMO_disp/hist12_{i}.csv')
    model12_final[i-1] = hist12.iloc[-1, :].to_numpy()
    model12.save(f'MIMO_disp/model12_{i}', save_format='h5')

    model13 = build_model(layer13, features, optim, lossfunc, metric)
    hist13 = model13.fit(x=Utrainset, y=fytrainset, batch_size=batchsize,
                         verbose=0, epochs=tr_epoch,
                         validation_data=(Utestset, fytestset))
    hist13 = pd.DataFrame(hist13.history, index=epoch)
    hist13.to_csv(f'MIMO_disp/hist13_{i}.csv')
    model13_final[i-1] = hist13.iloc[-1, :].to_numpy()
    model13.save(f'MIMO_disp/model13_{i}', save_format='h5')

    model14 = build_model(layer14, features, optim, lossfunc, metric)
    hist14 = model14.fit(x=Utrainset, y=fytrainset, batch_size=batchsize,
                         epochs=tr_epoch, verbose=0,
                         validation_data=(Utestset, fytestset))
    hist14 = pd.DataFrame(hist14.history, index=epoch)
    hist14.to_csv(f'MIMO_disp/hist14_{i}.csv')
    model14_final[i-1] = hist14.iloc[-1, :].to_numpy()
    model14.save(f'MIMO_disp/model14_{i}', save_format='h5')

    model15 = build_model(layer15, features, optim, lossfunc, metric)
    hist15 = model15.fit(x=Utrainset, y=fytrainset, batch_size=batchsize,
                         epochs=tr_epoch, verbose=0,
                         validation_data=(Utestset, fytestset))
    hist15 = pd.DataFrame(hist15.history, index=epoch)
    hist15.to_csv(f'MIMO_disp/hist15_{i}.csv')
    model15_final[i-1] = hist15.iloc[-1, :].to_numpy()
    model15.save(f'MIMO_disp/model15_{i}', save_format='h5')

    model16 = build_model(layer16, features, optim, lossfunc, metric)
    hist16 = model16.fit(x=Utrainset, y=fytrainset, batch_size=batchsize,
                         epochs=tr_epoch, verbose=0,
                         validation_data=(Utestset, fytestset))
    hist16 = pd.DataFrame(hist16.history, index=epoch)
    hist16.to_csv(f'MIMO_disp/hist16_{i}.csv')
    model16_final[i-1] = hist16.iloc[-1, :].to_numpy()
    model16.save(f'MIMO_disp/model16_{i}', save_format='h5')


# %% ===== Evaluate models =====

# model12_avg = np.mean(model12_final, axis=0)
# model13_avg = np.mean(model13_final, axis=0)
# model14_avg = np.mean(model14_final, axis=0)
# model15_avg = np.mean(model15_final, axis=0)
# model16_avg = np.mean(model16_final, axis=0)

# modellayers = np.arange(5, 12)
# relRMS_train_avg = np.array([model12_avg[1], model13_avg[1], model14_avg[1],
#                              model15_avg[1], model16_avg[1], model10_avg[1],
#                              model11_avg[1]])
# relRMS_val_avg = np.array([model12_avg[3], model13_avg[3], model14_avg[3],
#                            model15_avg[3], model16_avg[3], model10_avg[3],
#                            model11_avg[3]])

# sns.set(context='paper', style='whitegrid')
# plt.clf()
# plt.figure(1)
# plt.plot(modellayers, relRMS_train_avg, 'b', label='Avg rel_RMS on training')
# plt.plot(modellayers, relRMS_val_avg, 'r', label='Avg rel_RMS on validation')
# plt.xlabel('Model number of hidden layers', fontsize=18)
# plt.ylabel('Relative RMSE', fontsize=18)
# plt.suptitle('Average rel RMSE of trained MIMO models', fontsize=24)
# plt.legend()
# plt.show()

# %% ===== Save the final hist values ======
np.savetxt('model12_final.csv', model12_final, delimiter=',')
np.savetxt('model13_final.csv', model13_final, delimiter=',')
np.savetxt('model14_final.csv', model14_final, delimiter=',')
np.savetxt('model15_final.csv', model15_final, delimiter=',')
np.savetxt('model16_final.csv', model16_final, delimiter=',')
