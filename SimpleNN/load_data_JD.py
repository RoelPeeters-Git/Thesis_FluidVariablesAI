#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 17:31:30 2019

@author: Jan Decuyper
"""
import pandas as pd


data = pd.read_csv('Uy.csv')
data = data.values
#data = float(data)
#print(data)

X = data.T
Xtrain = X[:,1000:6000]
Xtest = X[:,6001:X.shape[1]]

Cy = pd.read_csv('Cy.csv')
Cy = Cy.values.T

Ytrain = Cy[:,1000:6000]
Ytest = Cy[:,6001:Cy.shape[1]]


