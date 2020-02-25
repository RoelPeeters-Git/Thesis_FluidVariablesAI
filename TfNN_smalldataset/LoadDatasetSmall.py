# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 14:31:19 2020

@author: Peeters Roel
"""
from pandas import read_csv
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
"""
Python script to obtain data from generated CSV-files
This code will extract the data and separate it in a training and testing batch

Inputs:
    Uy -- velocity y-component of the windtunnel data
    Cy -- force coëfficiënt y-component of the windtunnel data
    3 different datasets are used: singlesine input/output, multisine input/
    output and sweepsine input/output

Outputs:
    Xtrain, Xtest -- Numpy arrays from Uy separated into training and test data
    Ytrain, Ytest -- Numpy arrays from Cy separated into training and test data
    each dataset will have it's own train and test data
"""

Uy_single = read_csv('../DatasetSmall/f0p00Rel_relAmp0p00/Uy_Xtra.csv',
                         header=None).to_numpy()
Cy_single = read_csv('../DatasetSmall/f0p00Rel_relAmp0p00/Cy_Xtra.csv',
                         header=None).to_numpy()
Uy_multi = read_csv('../DatasetSmall/f0t1p5Rel_fRes0p1Abs_relAmp0p20_1/Uy_Xtra.csv',
                        header=None).to_numpy()
Cy_multi = read_csv('../DatasetSmall/f0t1p5Rel_fRes0p1Abs_relAmp0p20_1/Cy_Xtra.csv',
                        header=None).to_numpy()
Uy_sweep = read_csv('../DatasetSmall/SWS_fSt3p0_f0t1p5Rel_T16_relAmp0p20/Uy_Xtra.csv',
                  header=None).to_numpy()
Cy_sweep = read_csv('../DatasetSmall/SWS_fSt3p0_f0t1p5Rel_T16_relAmp0p20/Cy_Xtra.csv',
                  header=None).to_numpy()

Usingle_train, Usingle_test, Csingle_train, Csingle_test = train_test_split(
        Uy_single, Cy_single, test_size=0.2)

Umulti_train, Umulti_test, Cmulti_train, Cmulti_test = train_test_split(
        Uy_multi, Cy_multi, test_size=0.2)

Usweep_train, Usweep_test, Csweep_train, Csweep_test = train_test_split(
        Uy_sweep, Cy_sweep, test_size=0.2)

#plt.figure()
#plt.plot(Cy_single)
#plt.plot(Uy_single[:,1])

