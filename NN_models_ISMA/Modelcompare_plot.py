# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 13:36:10 2020

Script to compare the different averaged models and plot

@author: Peeters Roel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% ===== Import data =====

path = 'MIMO_disp/'             # MIMO model
# path = 'MISO_StandardModel/'    # MISO standard model
# path = 'MISO_disp/'             # MISO model with displacement
# path = 'MISO_shiftedInput/'     # MISO model with shifted input

end = 17         # 16 for standard model, 12 for other models, 17 for MIMO
Model = {}
for i in range(5, end):
    Model[f'model{i:02d}'] = pd.read_csv(path+f'Model{i}_final.txt',
                                         index_col=None,
                                         header=None).to_numpy()

avg = {}
for model, array in Model.items():
    avg[model] = np.nanmean(array, axis=0)

# %% ===== unpacking dict =====
models = np.array(list(avg.keys()))
values = np.array(list(avg.values()))

# %% ===== plot data =====
modellayers = np.arange(5, end)

sns.set(context='notebook', style='white')
plt.clf()
plt.figure(1)
plt.xticks(ticks=modellayers, labels=modellayers, fontsize=18)
plt.yticks(fontsize=18)
# plt.xlim(4.5, 15.5)
# plt.grid(True, which='both')
plt.plot(modellayers, values[:, 1], 'b*', label='Avg rel_RMS on training')
plt.plot(modellayers, values[:, 3], 'r*', label='Avg rel_RMS on validation')
plt.xlabel('Model number of layers', fontsize=22)
plt.ylabel('Relative RMSE', fontsize=22)
# plt.suptitle('Comparison of average relative RMSE of the trained models',
#              fontsize=28)
plt.show()
