# -*- coding: utf-8 -*-
# %% ===== Required packages =====

import pandas as pd
# import numpy as np
from GeneralFunctions import upgradedata

# %% ===== Docstring =====
"""
Created on Sat Mar 7 2020

@author: Peeters Roel
"""
"""
Masterthesis project: Determining Fluid Variables with AI

Preprocess the datasets into pandas dataframe ready to be converted to
tf Tensors. The datasets are split in training and test-sets.
"""
# %% ===== Data import =====

datapath = '../../DataFiles/Dataset_100Hz/relAmp0p05/'
X1 = pd.read_csv(datapath+'U_A0p05_100_1.csv', header=None,
                 index_col=0)
X2 = pd.read_csv(datapath+'U_A0p05_100_2.csv', header=None,
                 index_col=0)
X3 = pd.read_csv(datapath+'U_A0p05_100_3.csv', header=None,
                 index_col=0)
X4 = pd.read_csv(datapath+'U_A0p05_100_4.csv', header=None,
                 index_col=0)
X5 = pd.read_csv(datapath+'U_A0p05_100_5.csv', header=None,
                 index_col=0)
Y1 = pd.read_csv(datapath+'force_A0p05_100_1.csv', header=None,
                 index_col=0)
Y2 = pd.read_csv(datapath+'force_A0p05_100_2.csv', header=None,
                 index_col=0)
Y3 = pd.read_csv(datapath+'force_A0p05_100_3.csv', header=None,
                 index_col=0)
Y4 = pd.read_csv(datapath+'force_A0p05_100_4.csv', header=None,
                 index_col=0)
Y5 = pd.read_csv(datapath+'force_A0p05_100_5.csv', header=None,
                 index_col=0)

# %% ===== Data preprocessing =====
"""
Each Cy-value is represented by 3 timesteps of the Uy-measurements.
This is simulated by adding data of timesteps t-1, t-2 to each timestep t.
The first 2 timesteps are dropped.
"""

Xtrain = [upgradedata(f) for f in [X1, X2, X3, X4]]
Xtrain = pd.concat(Xtrain)
Xtest = upgradedata(X5)

Ytrain = [f.iloc[2:, :] for f in [Y1, Y2, Y3, Y4]]
Ytrain = pd.concat(Ytrain)
Ytest = Y5.iloc[2:, :]

Xtrain.to_csv('UyTrain_A0p05.csv')
Xtest.to_csv('UyTest_A0p05.csv')
Ytrain.to_csv('CyTrain_A0p05.csv')
Ytest.to_csv('CyTest_A0p05.csv')
