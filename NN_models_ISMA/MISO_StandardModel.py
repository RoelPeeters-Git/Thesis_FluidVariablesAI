# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 12:47:34 2020

Fully connected NN for MISO model of VIV CFD Data:
    Map of Uy and y as input to cy as output

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
ftrainset = pd.read_csv('ftrainset.csv', header=None).to_numpy()
Utestset = pd.read_csv('Utestset.csv', header=None).to_numpy()
ftestset = pd.read_csv('ftestset.csv', header=None).to_numpy()


# %% ===== Set the hyperparameters =====

path_to_save = 'MISO_StandardModel/'
features = len(Utrainset[1])
outputs = len(ftrainset[1])
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
layer5 = [16, 16, 8, 8, outputs]

layer6 = [16, 16, 8, 8, 4, outputs]

layer7 = [16, 16, 8, 8, 4, 4, outputs]

layer8 = [16, 16, 12, 8, 8, 4, 4, outputs]

layer9 = [16, 16, 12, 12, 8, 8, 4, 4, outputs]

layer10 = [16, 16, 16, 12, 12, 8, 8, 4, 4, outputs]

layer11 = [16, 16, 16, 12, 12, 12, 8, 8, 4, 4, outputs]

# %% ===== Train the neural networks =====
epoch = np.arange(1, tr_epoch+1)

model5_final = np.zeros((5, 4), dtype='float32')
model6_final = np.zeros((5, 4), dtype='float32')
model7_final = np.zeros((5, 4), dtype='float32')
model8_final = np.zeros((5, 4), dtype='float32')
model9_final = np.zeros((5, 4), dtype='float32')
model10_final = np.zeros((5, 4), dtype='float32')
model11_final = np.zeros((5, 4), dtype='float32')

for i in range(1, 6):

    model5 = build_model(layer5, features, optim, lossfunc, metric)
    hist5 = model5.fit(x=Utrainset, y=ftrainset, batch_size=batchsize,
                       epochs=tr_epoch, verbose=0,
                       validation_data=(Utestset, ftestset))
    hist5 = pd.DataFrame(hist5.history, index=epoch)
    hist5.to_csv(path_to_save+f'hist5_{i}.csv')
    model5_final[i-1] = hist5.iloc[-1, :].to_numpy()
    model5.save(path_to_save+f'Model5_{i}', save_format='h5')

    model6 = build_model(layer6, features, optim, lossfunc, metric)
    hist6 = model6.fit(x=Utrainset, y=ftrainset, batch_size=batchsize,
                       verbose=0, epochs=tr_epoch,
                       validation_data=(Utestset, ftestset))
    hist6 = pd.DataFrame(hist6.history, index=epoch)
    hist6.to_csv(path_to_save+f'hist6_{i}.csv')
    model6_final[i-1] = hist6.iloc[-1, :].to_numpy()
    model6.save(path_to_save+f'Model6_{i}', save_format='h5')

    model7 = build_model(layer7, features, optim, lossfunc, metric)
    hist7 = model7.fit(x=Utrainset, y=ftrainset, batch_size=batchsize,
                       epochs=tr_epoch, verbose=0,
                       validation_data=(Utestset, ftestset))
    hist7 = pd.DataFrame(hist7.history, index=epoch)
    hist7.to_csv(path_to_save+f'hist7_{i}.csv')
    model7_final[i-1] = hist7.iloc[-1, :].to_numpy()
    model7.save(path_to_save+f'Model7_{i}', save_format='h5')

    model8 = build_model(layer8, features, optim, lossfunc, metric)
    hist8 = model8.fit(x=Utrainset, y=ftrainset, batch_size=batchsize,
                       epochs=tr_epoch, verbose=0,
                       validation_data=(Utestset, ftestset))
    hist8 = pd.DataFrame(hist8.history, index=epoch)
    hist8.to_csv(path_to_save+f'hist8_{i}.csv')
    model8_final[i-1] = hist8.iloc[-1, :].to_numpy()
    model8.save(path_to_save+f'/Model8_{i}', save_format='h5')

    model9 = build_model(layer9, features, optim, lossfunc, metric)
    hist9 = model9.fit(x=Utrainset, y=ftrainset, batch_size=batchsize,
                       epochs=tr_epoch, verbose=0,
                       validation_data=(Utestset, ftestset))
    hist9 = pd.DataFrame(hist9.history, index=epoch)
    hist9.to_csv(path_to_save+f'hist9_{i}.csv')
    model9_final[i-1] = hist9.iloc[-1, :].to_numpy()
    model9.save(path_to_save+f'Model9_{i}', save_format='h5')

    model10 = build_model(layer10, features, optim, lossfunc, metric)
    hist10 = model10.fit(x=Utrainset, y=ftrainset, batch_size=batchsize,
                         epochs=tr_epoch, verbose=0,
                         validation_data=(Utestset, ftestset))
    hist10 = pd.DataFrame(hist10.history, index=epoch)
    hist10.to_csv(path_to_save+f'hist10_{i}.csv')
    model10_final[i-1] = hist10.iloc[-1, :].to_numpy()
    model10.save(path_to_save+f'Model10_{i}', save_format='h5')

    model11 = build_model(layer11, features, optim, lossfunc, metric)
    hist11 = model11.fit(x=Utrainset, y=ftrainset, batch_size=batchsize,
                         epochs=tr_epoch, verbose=0,
                         validation_data=(Utestset, ftestset))
    hist11 = pd.DataFrame(hist11.history, index=epoch)
    hist11.to_csv(path_to_save+f'hist11_{i}.csv')
    model11_final[i-1] = hist11.iloc[-1, :].to_numpy()
    model11.save(path_to_save+f'Model11_{i}', save_format='h5')

# %% ===== Evaluate models =====

model5_avg = np.mean(model5_final, axis=0)
model6_avg = np.mean(model6_final, axis=0)
model7_avg = np.mean(model7_final, axis=0)
model8_avg = np.mean(model8_final, axis=0)
model9_avg = np.mean(model9_final, axis=0)
model10_avg = np.mean(model10_final, axis=0)
model11_avg = np.mean(model10_final, axis=0)

modellayers = np.arange(5, 12)
relRMS_train_avg = np.array([model5_avg[1], model6_avg[1], model7_avg[1],
                             model8_avg[1], model9_avg[1], model10_avg[1],
                             model11_avg[1]])
relRMS_val_avg = np.array([model5_avg[3], model6_avg[3], model7_avg[3],
                           model8_avg[3], model9_avg[3], model10_avg[3],
                           model11_avg[3]])

sns.set(context='paper', style='whitegrid')
plt.clf()
plt.figure(1)
plt.plot(modellayers, relRMS_train_avg, 'b', label='Avg rel_RMS on training')
plt.plot(modellayers, relRMS_val_avg, 'r', label='Avg rel_RMS on validation')
plt.xlabel('Model number of hidden layers', fontsize=18)
plt.ylabel('Relative RMSE', fontsize=18)
plt.suptitle('Average rel RMSE of trained MISO models', fontsize=24)
plt.legend()
plt.show()

# %% ===== Save final hist ======

np.savetxt('Model5_final.csv', model5_final, delimiter=',')
np.savetxt('Model6_final.csv', model6_final, delimiter=',')
np.savetxt('Model7_final.csv', model7_final, delimiter=',')
np.savetxt('Model8_final.csv', model8_final, delimiter=',')
np.savetxt('Model9_final.csv', model9_final, delimiter=',')
np.savetxt('Model10_final.csv', model10_final, delimiter=',')
np.savetxt('Model11_final.csv', model11_final, delimiter=',')
