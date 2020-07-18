# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:58 2020

@author: Peeters Roel

Collect and process data for MISO models with:
    - U_y in input, c_y in output
    - U_y containing current timestep values and shifted timestep values


"""

import numpy as np
import pandas as pd

# %% ===== Import data to be processed =====
Utrain = pd.read_csv('Utrain.csv', index_col=None, header=None).to_numpy()
ftrain = pd.read_csv('ftrain.csv', index_col=None, header=None).to_numpy()
Utest = pd.read_csv('Utest.csv', index_col=None, header=None).to_numpy()
ftest = pd.read_csv('ftest.csv', index_col=None, header=None).to_numpy()

# %% ===== Process data =====

Utrain_original = Utrain.copy()
ftrain_original = ftrain.copy()

Utest_original = Utest.copy()
ftest_original = ftest.copy()

# Shift data based on correlation
shift = [14, 23, 14, 18, 14, 18, 23, 21, 23, 27,
         25, 27, 31, 29, 31, 4, 34, 4, 38, 37, 38]

### Training data ###
uShift = np.zeros(Utrain.shape)

for i in range(len(shift)):
    uShift[0:-shift[i], i] = Utrain[shift[i]:, i]

# concatenate original training set with shifted training set
Utrain = np.concatenate((Utrain_original, uShift), axis=1)

# Drop unknow input values
Utrain = Utrain[0:-np.max(shift), :]
ftrain = ftrain[0:-np.max(shift)]

# Mask uTrain to be able to split in batches
Utrain = Utrain[57:, :]
ftrain = ftrain[57:]  # N = 18900 -> batch size 189 or multiple (756)

### Validation data ###
uShift = np.zeros(Utest.shape)

for i in range(len(shift)):
    uShift[0:-shift[i], i] = Utest[shift[i]:, i]

Utest = np.concatenate((Utest_original, uShift), axis=1)

Utest = Utest[0:-np.max(shift), :]
ftest = ftest[0:-np.max(shift)]

# %% ===== Save datasets to csv =====

np.savetxt('Utrainshift.csv', Utrain, delimiter=',')
np.savetxt('ftrainshift.csv', ftrain, delimiter=',')
np.savetxt('Utestshift.csv', Utest, delimiter=',')
np.savetxt('ftestshift.csv', ftest, delimiter=',')
