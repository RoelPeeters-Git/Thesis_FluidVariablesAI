from pandas import read_csv
# import matplotlib.pyplot as plt

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 15:41:52 2019

@author: Peeters Roel
"""
"""
Python script to obtain data from generated CSV-files
This code will extract the data and separate it in a training and testing batch

Inputs:
    Uy -- velocity y-component of the windtunnel data
    Cy -- force coëfficiënt y-component of the windtunnel data

Outputs:
    Xtrain, Xtest -- Numpy arrays from Uy separated into training and test data
    Ytrain, Ytest -- Numpy arrays from Cy separated into training and test data
"""


def path_to_load(select_file):
    """
    Helper function to select the correct file in load_data function
    Parameters
    ----------
    select_file : int between 0 and 2

    Returns
    -------
    Tuple :
        strings to the paths that are selected
    """

    switcher = {
        1: ('../DatasetSmall/f0p00Rel_relAmp0p00/Uy_grid.csv',
            '../DatasetSmall/f0p00Rel_relAmp0p00/Cy_grid.csv'),
        2: ('../DatasetSmall/f0t1p5Rel_fRes0p1Abs_relAmp0p20_1/Uy_grid.csv',
            '../DatasetSmall/f0t1p5Rel_fRes0p1Abs_relAmp0p20_1/Cy_grid.csv'),
        3: ('../DatasetSmall/SWS_fSt3p0_f0t1p5Rel_T16_relAmp0p20/Uy_grid.csv',
            '../DatasetSmall/SWS_fSt3p0_f0t1p5Rel_T16_relAmp0p20/Cy_grid.csv')
        }
    return switcher.get(select_file, 'Invalid file')


def load_data(select_file):
    """
    Helper function to obtain data from the CSV-files to be loaded
    for NN-training

    Parameters
    ----------
    select_file : int
        between 0 and 2, to be passed to path_to_load function

    Returns
    -------
    Xtrain, Xtest, Ytrain, Ytest : numpy-array
        training and test data for NN-training
    """

    x_path = path_to_load(select_file)[0]
    y_path = path_to_load(select_file)[1]
    Xdata = read_csv(x_path)
    Ydata = read_csv(y_path)

    X = Xdata.values
    Y = Ydata.values
#   X = Xdata.T
#   Y = Ydata.T
    divider = round(Xdata.shape[0]*9.0/10.0)

    Xtrain = X[0:divider, :]
    Xtest = X[divider+1:X.shape[0], :]

    Ytrain = Y[0:divider, :]
    Ytest = Y[divider+1:Y.shape[0], :]

#   plt.plot(Ytrain.T)
#   plt.show()

    return Xtrain, Xtest, Ytrain, Ytest

# Xtr, Xte, Ytr, Yte = load_data(0)

# fig, axes = plt.subplots(2, 1)
# axes[0].plot(Xtr[:,0])
# axes[1].plot(Ytr)
# fig.show()
