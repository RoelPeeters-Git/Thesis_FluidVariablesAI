# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 14:33:22 2020

@author: Peeters Roel
"""

# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

SFhist = pd.read_csv('SFModel1_hist.csv', index_col=None)

# %% Plotting

plt.clf()
plt.figure(1)
plt.grid(True, which='both')
plt.semilogy(SFhist['Epoch'], SFhist['loss'], 'b',
             label='MSE on training data')
plt.semilogy(SFhist['Epoch'], SFhist['root_mean_squared_error'], 'r',
             label='RMSE on training data')
plt.ylim(1e-2, 2)
plt.xlabel('Training Epoch', fontsize=28)
plt.legend(loc='upper right', fontsize=28)
plt.suptitle('Evolution of training the neural network', fontsize=32)
