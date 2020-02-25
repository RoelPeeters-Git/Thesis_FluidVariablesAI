# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 12:02:39 2020

@author: Peeters Roel
"""
import pandas as pd
import numpy as np
""" 
Script to preprocess data from CSV-files into training data for TF-NN.
The data will be stripped of its transient behaviour.
Each line contains velocity at 21 probes in the wake of a cilinder
at timestamp t = linenumber.
Thereafter the data will be extended so that each line will contain 
the velocitydata at timestamp t, t-1 and t-2.
The new data will be used to calculate Cy = f(Uy(t), Uy(t-1), Uy(t-2)) and
will approach more to a NARX-model.

The new data is stored in a CSV-file, also the Cy-data is adapted to have 
the same shape[0] as the Uy-data.
"""

def upgrade_data(x):
    x_t2 = x[0:-2, :]
    x_t1 = x[1:-1, :]
    x_t = x[2:, :]

    output = np.concatenate((x_t, x_t1, x_t2), axis=1)
    assert(output.shape[0] == x.shape[0]-2)
    assert(output.shape[1] == x.shape[1]*3)

    return output

Uy = pd.read_csv('Uy_grid.csv', header=None).to_numpy(dtype='float32')
Cy = pd.read_csv('Cy_grid.csv', header=None).to_numpy(dtype='float32')

Uy = Uy[999:,:]
Cy = Cy[999:,:]

Uy_extended = upgrade_data(Uy)
Cy_extended = np.delete(Cy, [0,1], axis=0)

Uy_extended = pd.DataFrame(Uy_extended)
Uy_extended.to_csv('Uy_extended.csv', header=None, index=False)
Cy_extended = pd.DataFrame(Cy_extended)
Cy_extended.to_csv('Cy_extended.csv', header=None, index=False)

