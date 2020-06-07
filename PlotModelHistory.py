# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# import seaborn as sns

"""
Created on Fri May 29 12:57:03 2020

@author: Peeters Roel

Masterthesis project:

    This script will create graphs to include in the thesis, illustrating the
    performance of the trained NN-models
"""

# %% ===== Obtaining the correct history files =====

path = 'C://users/peete/VUB/Masterthesis/Code/'
# history_path = path + 'NN_Model_SF/SFModel1_hist.csv'
# history_path = path + 'NN_Model_SF/SFModel2_hist.csv'
history_path = path + 'NN_Model_AugmInput/AugmModel1_hist.csv'
# history_path = path + 'NN_Model_AugmInput/AugmModel2_hist.csv'

history = pd.read_csv(history_path, index_col=0)
epochs = np.arange(1, len(history)+1)
history.insert(0, 'Epochs', epochs)

# headers = list(history.columns)
# histvalues = {}
# for header in headers:
#     histvalues[header] = history[header].to_numpy()


# %% ===== Plot loss data =====

figloss, axloss = plt.subplots()
axloss.set(xscale='linear', yscale='log')
axloss.xaxis.grid(True, which='major')
axloss.yaxis.grid(True, which='both')
axloss.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
yformatter = mpl.ticker.LogFormatter(labelOnlyBase=False,
                                     minor_thresholds=(3, 1))
axloss.yaxis.set_minor_formatter(yformatter)
axloss.yaxis.set_major_formatter(yformatter)
axloss.set_xlabel('Epoch')
axloss.set_ylabel('Loss')
axloss.plot(history['Epochs'], history['loss'], 'b', label='Training set loss')
axloss.plot(history['Epochs'], history['val_loss'], 'r',
            label='Validation set Loss')
axloss.legend()
figloss.suptitle('Evolution of Loss per epoch')
figloss.show()

# %% ===== Plot RMSE data =====

figrmse, axrmse = plt.subplots()
axrmse.set(xscale='linear', yscale='log', xlabel='Epoch', ylabel='RMSE')
axrmse.xaxis.grid(True, which='major')
axrmse.yaxis.grid(True, which='both')
axrmse.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

# yformatter = mpl.ticker.ScalarFormatter(useOffset=True)
# yformatter.set_scientific(False)
axrmse.yaxis.set_minor_formatter(yformatter)
axrmse.yaxis.set_major_formatter(yformatter)
axrmse.plot(history['Epochs'], history['root_mean_squared_error'],
            'b', label='Root Mean Squared Error')
axrmse.plot(history['Epochs'], history['val_root_mean_squared_error'],
            'r', label='Validation Root Mean Squared Error')
axrmse.legend()
figrmse.suptitle('Evolution of RMSE per Epoch')
figrmse.show()
