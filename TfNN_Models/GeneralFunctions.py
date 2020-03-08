# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
"""
Created on Sat Mar  7 14:15:58 2020

@author: Peeters Roel
"""
"""
Module: preprocess data by extending the inputdata
"""


def upgradedata(x):
    idx = x.index
    x = x.to_numpy()
    x_t2 = x[0:-2]
    x_t1 = x[1:-1]
    x_t = x[2:]
    output = np.concatenate((x_t, x_t1, x_t2), axis=1)
    return pd.DataFrame(output, index=idx[2:])


# x = pd.DataFrame([[1, 2, 3],
#                   [4, 5, 6],
#                   [7, 8, 9],
#                   [10, 11, 12]])
# y = pd.DataFrame([[7, 8, 9, 4, 5, 6, 1, 2, 3],
#                   [10, 11, 12, 7, 8, 9, 4, 5, 6]], index=(2, 3))

# x1 = upgradedata(x)

# x_t2 = x.iloc[0:-2, :]
# x_t1 = x.iloc[1:-1, :]
# x_t = x.iloc[2:, :]
