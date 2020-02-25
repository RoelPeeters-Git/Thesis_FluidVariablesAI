# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

"""
Created on Mon Dec 23 11:51:26 2019

@author: Peeters Roel

A helper function that will enhance the simulation data for better input
into a DNN. Velocitydata from a csv-file is imported.
Each line contains velocity at 21 probes in the wake of a cilinder
at timestamp t = linenumber.
The data will be enhanced so that each line will contain the velocitydata at
timestamp t, t-1 and t-2.
The new data will be used to calculate Cy = f(Uy(t), Uy(t-1), Uy(t-2)).

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


Uy_singlesine = pd.read_csv('f0p00Rel_relAmp0p00/Uy_grid.csv', 
                           header=None).to_numpy()
Cy_singlesine = pd.read_csv('f0p00Rel_relAmp0p00/Cy_grid.csv', 
                           header=None).to_numpy()

Uy_multisine = pd.read_csv('f0t1p5Rel_fRes0p1Abs_relAmp0p20_1/Uy_grid.csv',
                          header=None).to_numpy()
Cy_multisine = pd.read_csv('f0t1p5Rel_fRes0p1Abs_relAmp0p20_1/Cy_grid.csv',
                          header=None).to_numpy()

Uy_sws = pd.read_csv('SWS_fSt3p0_f0t1p5Rel_T16_relAmp0p20/Uy_grid.csv', 
                    header=None).to_numpy()
Cy_sws = pd.read_csv('SWS_fSt3p0_f0t1p5Rel_T16_relAmp0p20/Cy_grid.csv', 
                    header=None).to_numpy()

Uy_singlesine = upgrade_data(Uy_singlesine)
Cy_singlesine = np.delete(Cy_singlesine, [0,1], axis=0)
assert(Uy_singlesine.shape[0]==Cy_singlesine.shape[0])

Uy_multisine = upgrade_data(Uy_multisine)
Cy_multisine = np.delete(Cy_multisine, [0,1], axis=0)
assert(Uy_multisine.shape[0]==Cy_multisine.shape[0])

Uy_sws = upgrade_data(Uy_sws)
Cy_sws = np.delete(Cy_sws, [0,1], axis=0)
assert(Uy_sws.shape[0]==Cy_sws.shape[0])

pd.DataFrame(Uy_singlesine).to_csv('f0p00Rel_relAmp0p00/Uy_Xtra.csv', 
            header=None, index=None)
pd.DataFrame(Cy_singlesine).to_csv('f0p00Rel_relAmp0p00/Cy_Xtra.csv',
            header=None, index=None)

pd.DataFrame(Uy_multisine).to_csv('f0t1p5Rel_fRes0p1Abs_relAmp0p20_1/Uy_Xtra.csv', 
            header=None, index=None)
pd.DataFrame(Cy_multisine).to_csv('f0t1p5Rel_fRes0p1Abs_relAmp0p20_1/Cy_Xtra.csv',
            header=None, index=None)

pd.DataFrame(Uy_sws).to_csv('SWS_fSt3p0_f0t1p5Rel_T16_relAmp0p20/Uy_Xtra.csv', 
            header=None, index=None)
pd.DataFrame(Cy_sws).to_csv('SWS_fSt3p0_f0t1p5Rel_T16_relAmp0p20/Cy_Xtra.csv',
            header=None, index=None)

# x = np.array([[1, 2, 3],
#               [4, 5, 6],
#               [7, 8, 9],
#               [10, 11, 12]])

# y = upgrade_data(x)
# print(y)