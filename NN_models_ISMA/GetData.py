# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 10:52:14 2020

@author: Peeters Roel

Collect and process data for MISO and MIMO models with:
    - U_y and Y in input, c_y in output
    - U_y in input, c_y and Y in output

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% ===== Collect datasets =====

path = 'C:/Users/peete/VUB/Masterthesis/DataFiles/Dataset_100Hz/SWS/'

files = os.listdir(path)

fData = {}
uData = {}
yData = {}

for f in files:
    f_name, f_ext = os.path.splitext(f)
    f_type, f_SWS, f_Amp, _ = f_name.split('_')

    if f_type == 'force':
        fData[f'{f_type}_{f_Amp}'] = pd.read_csv(path+f, index_col=0,
                                                 header=0).to_numpy()
    elif f_type == 'U':
        uData[f'{f_type}_{f_Amp}'] = pd.read_csv(path+f, index_col=0,
                                                 header=0).to_numpy()
    elif f_type == 'y':
        yData[f'{f_type}_{f_Amp}'] = pd.read_csv(path+f, index_col=0,
                                                 header=0).to_numpy()
    else:
        pass

# print(type(yData['y_A0p05']))
# %% ===== Process datasets ======

yData['y_A0p05'] = yData['y_A0p05'] * 0.05
yData['y_A0p10'] = yData['y_A0p10'] * 0.10
yData['y_A0p15'] = yData['y_A0p15'] * 0.15
yData['y_A0p20'] = yData['y_A0p20'] * 0.20
yData['y_A0p25'] = yData['y_A0p25'] * 0.25
yData['y_A0p30'] = yData['y_A0p30'] * 0.30

# Get input datasets with velocity and displacement
UyA0p05 = np.concatenate((uData['U_A0p05'], yData['y_A0p05']), axis=1)
UyA0p10 = np.concatenate((uData['U_A0p10'], yData['y_A0p10']), axis=1)
UyA0p15 = np.concatenate((uData['U_A0p15'], yData['y_A0p15']), axis=1)
UyA0p20 = np.concatenate((uData['U_A0p20'], yData['y_A0p20']), axis=1)
UyA0p25 = np.concatenate((uData['U_A0p25'], yData['y_A0p25']), axis=1)
UyA0p30 = np.concatenate((uData['U_A0p30'], yData['y_A0p30']), axis=1)

Uytrainset = np.concatenate((UyA0p05, UyA0p10, UyA0p15, UyA0p25, UyA0p30),
                            axis=0).astype('float32')
ftrainset = np.concatenate((fData['force_A0p05'], fData['force_A0p10'],
                            fData['force_A0p15'], fData['force_A0p25'],
                            fData['force_A0p30']), axis=0).astype('float32')
assert len(Uytrainset) == len(ftrainset)

Uytestset = UyA0p20.astype('float32')
ftestset = fData['force_A0p20'].astype('float32')
assert len(Uytestset) == len(ftestset)


# Get output datasets with force and displacement
fyA0p05 = np.concatenate((fData['force_A0p05'], yData['y_A0p05']), axis=1)
fyA0p10 = np.concatenate((fData['force_A0p10'], yData['y_A0p10']), axis=1)
fyA0p15 = np.concatenate((fData['force_A0p15'], yData['y_A0p15']), axis=1)
fyA0p20 = np.concatenate((fData['force_A0p20'], yData['y_A0p20']), axis=1)
fyA0p25 = np.concatenate((fData['force_A0p25'], yData['y_A0p25']), axis=1)
fyA0p30 = np.concatenate((fData['force_A0p30'], yData['y_A0p30']), axis=1)

Utrainset = np.concatenate((uData['U_A0p05'], uData['U_A0p10'],
                            uData['U_A0p15'], uData['U_A0p25'],
                            uData['U_A0p30']), axis=0).astype('float32')
fytrainset = np.concatenate((fyA0p05, fyA0p10, fyA0p15, fyA0p25, fyA0p30),
                            axis=0).astype('float32')

assert len(Utrainset) == len(fytrainset)

Utestset = uData['U_A0p20'].astype('float32')
fytestset = fyA0p20.astype('float32')

assert len(Utestset) == len(fytestset)

# %% ===== Save datasets =====
# np.savetxt('Uytrainset.csv', Uytrainset, delimiter=',')
# np.savetxt('ftrainset.csv', ftrainset, delimiter=',')
# np.savetxt('Uytestset.csv', Uytestset, delimiter=',')
# np.savetxt('ftestset.csv', ftestset, delimiter=',')

np.savetxt('Utrainset.csv', Utrainset, delimiter=',')
np.savetxt('fytrainset.csv', fytrainset, delimiter=',')
np.savetxt('Utestset.csv', Utestset, delimiter=',')
np.savetxt('fytestset.csv', fytestset, delimiter=',')

# %% ===== Plot displacement data =====

# sns.set(context='notebook', style='white')
# plt.figure(1)

# # ytrain = np.array(yData['y_A0p05'])
# # ytrain = np.append(ytrain, yData['y_A0p10'])
# # ytrain = np.append(ytrain, yData['y_A0p15'])
# # ytrain = np.append(ytrain, yData['y_A0p25'])
# # ytrain = np.append(ytrain, yData['y_A0p30'])

# tick_y = np.arange(-0.30, 0.40, 0.1)
# # label_y = np.arange(-3, 4, 2)

# subfig = 1
# for key, val in yData.items():
#     plt.subplot(3, 2, subfig)
#     plt.ylim(ymin=-0.35, ymax=0.35)
#     plt.yticks(ticks=tick_y)
#     plt.plot(val, '--', label=key)
#     if subfig == 3:
#         plt.ylabel('Imposed displacement y of the cylinder', fontsize=22)
#     if subfig == 5 or 6:
#         plt.xlabel('Datapoint', fontsize=22)
#     subfig += 1

# plt.show()
