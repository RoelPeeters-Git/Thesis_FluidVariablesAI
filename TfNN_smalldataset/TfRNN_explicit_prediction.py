# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:01:17 2020

@author: Peeters Roel
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from LoadDatasetSmall import Usingle_test, Csingle_test
from LoadDatasetSmall import Umulti_test, Cmulti_test
from LoadDatasetSmall import Usweep_test, Csweep_test

"""
Masterthesis project: Determining Fluid Variables with AI
Part 3

This project will use the Tensorflow framework to train and utilize a
deep neural network.
The DNN utilizes simulation data from a CFD-model of airflow around a cilinder.
Input features are velocity-vectors in points in the wake of the cilinder.
The output desired is the force acting on the cilinder.
In first instance only the y-component features will be utilized.

The input features have been modified, so that each Cy(n)-value is represented 
with Uy(n)-values of Uy(n), Uy(n-1) and Uy(n-2).
This simulates the dependency of Cy(n) to the past values of the velocityfield.
The NN will be verified with a true Tf-RNN in order to compare which method 
provides better results

This script uses a Tensorflow trained model to test the model and obtain 
comparison to the actual outputvalues.
"""

#%% === Evaluate the model on single sine wave dataset ===
modelsingle = tf.keras.models.load_model('modelsingle')
loss, acc = modelsingle.evaluate(Usingle_test, Csingle_test)
Yhat_single = modelsingle.predict(Usingle_test)

plt.figure(1)
plt.subplot(2,1,1)
plt.plot(Yhat_single, 'b:', Csingle_test, 'r--')
plt.subplot(2,1,2)
plt.plot(Yhat_single-Csingle_test)
plt.show()

#%% === Evaluate both models on multi sinewave dataset ===
modelmulti1 = tf.keras.models.load_model('modelmulti1')
loss_multi1, acc_multi1 = modelmulti1.evaluate(Umulti_test, Cmulti_test)
Yhat_multi1 = modelmulti1.predict(Umulti_test)
error_multi1 = Yhat_multi1 - Cmulti_test

modelmulti2 = tf.keras.models.load_model('modelmulti2')
loss_multi2, acc_multi2 = modelmulti2.evaluate(Umulti_test, Cmulti_test)
Yhat_multi2 = modelmulti2.predict(Umulti_test)
error_multi2 = Yhat_multi2 - Cmulti_test

modelcompare_multi = [loss_multi1-loss_multi2, acc_multi1-acc_multi2]

plt.figure(2)
plt.subplot(3,1,1)
plt.plot(Yhat_multi1[1380:1400], 'b:', Cmulti_test[1380:1400], 'r--')
plt.subplot(3,1,3)
plt.plot(error_multi1[1380:1400], 'g--', error_multi2[1380:1400], 'b--')
plt.subplot(3,1,2)
plt.plot(Yhat_multi2[1380:1400], 'b:', Cmulti_test[1380:1400], 'r--')

