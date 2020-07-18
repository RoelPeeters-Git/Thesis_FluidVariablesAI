# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 10:57:12 2020

@author: peete
"""
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np

# %% ===== Building functions =====


def rel_rmse(y_true, y_pred):
    out = tf.math.sqrt(
        tf.math.reduce_mean(
            tf.math.square(y_pred-y_true))) / tf.math.sqrt(
                tf.math.reduce_mean(tf.math.square(y_true)))
    return out


def rel_rms_np(y_true, y_sim):
    # Computes the relative root-mean-squared error using the Numpy library
    return (np.sqrt(np.mean(np.square(y_true-y_sim))) /
            np.sqrt(np.mean(np.square(y_true))))


def build_model(layers, features):
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

    inputs = keras.Input(shape=(features,))
    x = Dense(layers[0], activation='relu', name='Input')(inputs)
    y = Dense(layers[1], activation='relu', name='Force_input')(x)
    z = Dense(layers[1], activation='relu', name='Disp_input')(x)
    count = 1

    for layer in layers[2:]:
        y = Dense(layer, activation='relu', name=f'Force{count}')(y)
        count += 1

    out_force = Dense(1, name='output_force')(y)
    
    count = 1
    
    for layer in layers[2:]:
        z = Dense(layer, activation='relu', name=f'Displacement{count}')(z)
        count += 1

    out_y = Dense(1, name='output_displacement')(z)

    model = keras.Model(inputs=inputs, outputs=[out_force, out_y],
                        name=f'Model{len(layers)}')
    model.summary()

    return model


# %% ===== Import data and set variables ======

Utrainset = pd.read_csv('Utrainset.csv', header=None).to_numpy()
fytrainset = pd.read_csv('fytrainset.csv', header=None).to_numpy()
Utestset = pd.read_csv('Utestset.csv', header=None).to_numpy()
fytestset = pd.read_csv('fytestset.csv', header=None).to_numpy()

ftrainset = fytrainset[:, 0]
ytrainset = fytrainset[:, 1]
ftestset = fytestset[:, 0]
ytestset = fytestset[:, 1]

features = len(Utrainset[1])
optim = keras.optimizers.Adam()
lossfunc = keras.losses.MeanSquaredError()
metric = [rel_rmse, rel_rmse]
tr_epoch = 2000
batchsize = 655
epoch = np.arange(1, tr_epoch+1)

# %% ===== Define model layers =====

layer12 = [16, 16, 16, 12, 12, 12, 8, 8, 8, 4, 4]
# 11 layers + 1 defined in build_model function

model12 = build_model(layer12, features)
keras.utils.plot_model(model12, 'MIMO_model12.png', show_shapes=True)

# %% ===== Compile and train =====
model12.compile(optimizer=optim, loss=[lossfunc, lossfunc],
                metrics=metric)

history12 = model12.fit(Utrainset, {'output_force': ftrainset,
                                    'output_displacement': ytrainset},
                        batch_size=batchsize, epochs=tr_epoch,
                        validation_data=[Utestset,
                                         {'output_force': ftestset,
                                          'output_displacement': ytestset}])

hist12 = pd.DataFrame(history12.history, index=epoch)
model12.save('MIMO_model12')

# %% ===== Evaluate model =====

eval_model12 = model12.evaluate(Utestset, [ftestset, ytestset])
fpred, ypred = model12.predict(Utestset)

# %% ===== Calculate rel_RMS =====
relRMS_fpred = rel_rms_np(ftestset, fpred)
relRMS_ypred = rel_rms_np(ytestset, ypred)
relRMS_ypred_tf = rel_rmse(ytestset, ypred).numpy()
