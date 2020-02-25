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

from pandas import read_csv
import matplotlib.pyplot as plt

def load_data():
    Xdata = read_csv('f0p00Rel_relAmp0p00/Uy_grid.csv')
    Ydata = read_csv('f0p00Rel_relAmp0p00/Cy_grid.csv')
    
    Xdata = Xdata.values
    Ydata = Ydata.values
    X = Xdata.T
    Y = Ydata.T
    
    Xtrain = X[:,999:5999]
    Xtest = X[:,6000:X.shape[1]]
    
    Ytrain = Y[:,999:5999]
    Ytest = Y[:,6000:Y.shape[1]]
    
    plt.plot(Ytrain.T)
    plt.show()
    
    return Xtrain, Xtest, Ytrain, Ytest
