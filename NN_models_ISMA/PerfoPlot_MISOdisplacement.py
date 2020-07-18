# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 13:15:45 2020

Calculating the performance of the MIMO model
Plotting the history of the MIMO model

@author: Peeters Roel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


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


# Linear model estimation
def linear_reg_check(regressor, output):
    # Estimates a linear regression model in least-squares sense
    regressor = np.concatenate(
        (regressor, np.ones((output.shape[0], 1))), axis=1)
    # add vector of ones to allow for a bias term
    theta = (np.dot(np.dot(
        np.linalg.pinv(np.dot(regressor.T, regressor)), regressor.T), output))
    output_lin = np.dot(regressor, theta)
    return rel_rms_np(output, output_lin), theta


# %% ===== Get data =====

hist = pd.read_csv('MISO_disp/Model11/hist11_2.txt', index_col=0)


Uytrainset = pd.read_csv('Uytrainset.csv', header=None).to_numpy()
ftrainset = pd.read_csv('ftrainset.csv', header=None).to_numpy()
Uytestset = pd.read_csv('Uytestset.csv', header=None).to_numpy()
ftestset = pd.read_csv('ftestset.csv', header=None).to_numpy()


# %% ===== Plot history data =====
# plt.clf()

plt.figure(1)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.semilogy(hist.index, hist['rel_rmse'], 'b',
             subsy=[2, 3, 4, 5, 6, 7, 8, 9],
             label='Relative Rmse on training data')
plt.semilogy(hist.index, hist['val_rel_rmse'], 'r',
             subsy=[2, 3, 4, 5, 6, 7, 8, 9],
             label='Relative Rmse on validation data')
plt.xlabel('Training Epoch', fontsize=22)
plt.ylabel('Relative RMSE', fontsize=22)
plt.grid(True, which='both', axis='both')
plt.legend()
plt.suptitle('Evolution of training the model', fontsize=28)
# plt.figure(1)
# plt.plot(hist5.index, hist5['rel_rmse'], 'c')
# plt.plot()
plt.show()


# %% ===== Select best performing model =====

metric = {'rel_rmse': rel_rmse}
model = tf.keras.models.load_model('MISO_disp/Model11/Model11_2',
                                   custom_objects=metric)
tf.keras.utils.plot_model(model, 'MISO_dispModel.png', show_layer_names=True,
                          show_shapes=True)

# %% ===== Perform check of linear regression =====

train_relRMS_linear, theta = linear_reg_check(Uytrainset, ftrainset)

Uytestset_plus = np.concatenate((Uytestset, np.ones((Uytestset.shape[0], 1))),
                                axis=1)
fpred_linear = np.dot(Uytestset_plus, theta)

test_relRMS_linear = rel_rms_np(ftestset, fpred_linear)

# %% ===== Use TF model to predict and calculate performance =====

time = np.arange(0, (len(ftestset))/100, step=0.01)

modeleval = model.evaluate(Uytestset, ftestset)
fpred = model.predict(Uytestset)
assert fpred.shape == ftestset.shape
test_relRMS_model = rel_rms_np(ftestset, fpred)

force_error = ftestset - fpred

feval = model.predict(Uytrainset)
tr_relRMS_model = rel_rms_np(ftrainset, feval)

# %% ===== Save performance parameters =====
perfdata = {'Relative RMSE on training data': tr_relRMS_model,
            'Relative RMSE on test data': test_relRMS_model,
            'Linear test relative RMSE on training data':
                train_relRMS_linear,
            'Linear test relative RMSE on test data':
                test_relRMS_linear,
            }

perfdata = pd.DataFrame(perfdata, index=['Value']).transpose()
perfdata.to_csv('MISO_performancedata.txt')

# %% ===== Make plots ======
sns.set(context='notebook', style='white')
# plt.figure(2)
# plt.subplot(211)
# plt.plot(time, ftestset[:, 1], 'b--', label='True displacement')
# plt.ylabel('Displacement y', fontsize=18)
# plt.subplot(212)
# plt.plot(time, fpred[:, 1], 'r--', label='Predicted displacement')
# plt.plot(time, disp_error, 'k', label='Error of the predictions')
# plt.xlabel('Time (s)', fontsize=18)
# plt.ylabel('Displacement y / Error', fontsize=18)
# plt.suptitle('Displacement of the cylinder (y-direction)', fontsize=24)
# plt.show()

plt.figure(3)
plt.subplot(211)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.plot(time, ftestset, 'b--', label='True force coefficient')
plt.ylabel('Force coefficient', fontsize=20)
plt.subplot(212)
plt.plot(time, fpred, 'r--', label='Predicted force coefficient')
plt.plot(time, force_error, 'k', label='Error of the predictions')
plt.xlabel('Time (s)', fontsize=22)
plt.ylabel('Force coefficient / Error', fontsize=20)
# plt.suptitle('Force coefficient on the the cylinder (y-direction)',
#            fontsize=28)
plt.show()
