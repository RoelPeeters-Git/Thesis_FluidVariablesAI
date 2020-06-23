# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 17:12:31 2020

@author: Peeters Roel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Utest = pd.read_csv('Utest.csv', index_col=None, header=None)
ftest = pd.read_csv('ftest.csv', index_col=None, header=None)

time = np.arange(1, len(ftest)+1, 1)

sns.set(context='notebook', style='whitegrid')
plt.clf()
plt.figure(1)
plt.plot(time, Utest[8])
plt.xlabel('Datapunt', fontsize=24)
plt.ylabel('Snelheid U_y', fontsize=24)
plt.suptitle('Evolutie van de snelheid achter de cilinder', fontsize=32)

plt.figure(2)
plt.plot(time, ftest)
plt.xlabel('Datapunt', fontsize=24)
plt.ylabel('Krachtcoëfficiënt c_y', fontsize=24)
plt.suptitle('Evolutie van de kracht op de cilinder', fontsize=32)
