# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
"""
Created on Thu Jan 30 12:02:39 2020

@author: Peeters Roel

Script to preprocess data from CSV-files into training data for an augmented
input into a TF-NN.
Each line contains velocity at 21 probes in the wake of a cilinder
at timestamp t = linenumber.
Thereafter the input data will be extended so that each line will contain
the velocitydata at timestamp t, t-1 and t-2.
The new data will be used to calculate Cy = f(Uy(t), Uy(t-1), Uy(t-2)) and
will approach more to a NARX-model.
The SWS data is combined together and a training and test data set is
generated. This dataset will be saved as a CSV-file and will then be used
to train a NN model.

"""


def augment_input(x, y):
    x_t2 = x[0:-2, :]
    x_t1 = x[1:-1, :]
    x_t = x[2:, :]

    x_out = np.concatenate((x_t, x_t1, x_t2), axis=1)
    assert(x_out.shape[0] == x.shape[0]-2)
    assert(x_out.shape[1] == x.shape[1]*3)
    y_out = np.delete(y, [0, 1], axis=0)
    assert(y_out.shape[0] == x_out.shape[0])

    return x_out, y_out


# ===== Loading the required datasets =====
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

# ===== Datasets are processed to augment input data =====

U_amp0p05, f_amp0p05 = augment_input(U_amp0p05, f_amp0p05)
U_amp0p10, f_amp0p10 = augment_input(U_amp0p10, f_amp0p10)
U_amp0p15, f_amp0p15 = augment_input(U_amp0p15, f_amp0p15)
U_amp0p20, f_amp0p20 = augment_input(U_amp0p20, f_amp0p20)
U_amp0p25, f_amp0p25 = augment_input(U_amp0p25, f_amp0p25)
U_amp0p30, f_amp0p30 = augment_input(U_amp0p30, f_amp0p30)


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
