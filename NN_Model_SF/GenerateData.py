# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


"""
Created on Thu May  7 18:25:03 2020

@author: Peeters Roel

Masterthesis project

The SWS data is combined together and a training and test data set is
generated. This dataset will be saved as a CSV-file and will then be used
to train a NN model.
"""

path_to_data = 'C://Users/peete/VUB/Masterthesis/DataFiles/Dataset_100Hz/SWS/'

U_amp0p05 = pd.read_csv(path_to_data+'U_SWS_A0p05_100.csv',
                        index_col=0).to_numpy()
U_amp0p10 = pd.read_csv(path_to_data+'U_SWS_A0p10_100.csv',
                        index_col=0).to_numpy()
U_amp0p15 = pd.read_csv(path_to_data+'U_SWS_A0p15_100.csv',
                        index_col=0).to_numpy()
U_amp0p20 = pd.read_csv(path_to_data+'U_SWS_A0p20_100.csv',
                        index_col=0).to_numpy()
U_amp0p25 = pd.read_csv(path_to_data+'U_SWS_A0p25_100.csv',
                        index_col=0).to_numpy()
U_amp0p30 = pd.read_csv(path_to_data+'U_SWS_A0p30_100.csv',
                        index_col=0).to_numpy()

f_amp0p05 = pd.read_csv(path_to_data+'force_SWS_A0p05_100.csv',
                        index_col=0).to_numpy()
f_amp0p10 = pd.read_csv(path_to_data+'force_SWS_A0p10_100.csv',
                        index_col=0).to_numpy()
f_amp0p15 = pd.read_csv(path_to_data+'force_SWS_A0p15_100.csv',
                        index_col=0).to_numpy()
f_amp0p20 = pd.read_csv(path_to_data+'force_SWS_A0p20_100.csv',
                        index_col=0).to_numpy()
f_amp0p25 = pd.read_csv(path_to_data+'force_SWS_A0p25_100.csv',
                        index_col=0).to_numpy()
f_amp0p30 = pd.read_csv(path_to_data+'force_SWS_A0p30_100.csv',
                        index_col=0).to_numpy()

# ===== Create and store the training, dev and test set =====

# Devset and testset is created from the amp0p20 datasets, with devset from
# example 0 to 1749 and trainset from 1750 to 3100

Utrain = np.concatenate((U_amp0p05, U_amp0p10, U_amp0p15,
                         U_amp0p25, U_amp0p30), axis=0).astype('float32')
ftrain = np.concatenate((f_amp0p05, f_amp0p10, f_amp0p15,
                         f_amp0p25, f_amp0p30), axis=0).astype('float32')

Udev = U_amp0p20[500:3000, :].astype('float32')
fdev = f_amp0p20[500:3000, :].astype('float32')
Utest = U_amp0p20.astype('float32')
ftest = f_amp0p20.astype('float32')

np.savetxt('Utrain.csv', Utrain, delimiter=',')
np.savetxt('ftrain.csv', ftrain, delimiter=',')
np.savetxt('Udev.csv', Udev, delimiter=',')
np.savetxt('fdev.csv', fdev, delimiter=',')
np.savetxt('Utest.csv', Utest, delimiter=',')
np.savetxt('ftest.csv', ftest, delimiter=',')

# %% ===== Visualizing the force coefficients of each dataset =====
sns.set(style='whitegrid')
ylimits = [np.min(f_amp0p25), np.abs(np.min(f_amp0p25))]

fig, axs = plt.subplots(6, 1)
axs[0].plot(f_amp0p05, color='b', label='Amp0.05')
axs[1].plot(f_amp0p10, color='k', label='Amp0.10')
axs[2].plot(f_amp0p15, color='r', label='Amp0.15')
axs[3].plot(f_amp0p20, color='g', label='Amp0.20')
axs[4].plot(f_amp0p25, color='m', label='Amp0.25')
axs[5].plot(f_amp0p30, color='c', label='Amp0.30')

lines = []
labels = []

for ax in axs.flat:
    ax.set(xlabel='Sample Number', ylabel='C_y')
    ax.set_ylim(bottom=(ylimits[0]-0.2), top=(ylimits[1]+0.2))
    ax.legend(loc='upper right')
    axLine, axLabel = ax.get_legend_handles_labels()
    lines.extend(axLine)
    labels.extend(axLabel)
    ax.label_outer()
    ax.grid(True)

fig.suptitle('Force coefficient of all datasets', fontsize=14)
plt.show()

fig2, ax2 = plt.subplots(3, 2)
ax2[0, 0].plot(U_amp0p05[:, 8], color='b', label='Amp0.05')
ax2[0, 1].plot(U_amp0p10[:, 8], color='k', label='Amp0.10')
ax2[1, 0].plot(U_amp0p15[:, 8], color='r', label='Amp0.15')
ax2[1, 1].plot(U_amp0p20[:, 8], color='g', label='Amp0.20')
ax2[2, 0].plot(U_amp0p25[:, 8], color='m', label='Amp0.25')
ax2[2, 1].plot(U_amp0p30[:, 8], color='c', label='Amp0.30')

lines2 = []
labels2 = []

for ax in ax2.flat:
    ax.set(xlabel='Sample Number', ylabel='U_y')
    ax.set_ylim(-1.8, 1.8)
    ax.legend(loc='upper right')
    axLine, axLabel = ax.get_legend_handles_labels()
    lines2.extend(axLine)
    labels2.extend(axLabel)
    ax.label_outer()

fig2.suptitle('Velocity (y-component) in 1 probe of all datasets', fontsize=14)
fig2.show()
