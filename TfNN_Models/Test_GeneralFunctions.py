# -*- coding: utf-8 -*-
import pandas as pd
from GeneralFunctions import upgradedata
"""
Created on Sat Mar  7 16:22:51 2020

@author: Peeters Roel
"""
"""
Module: tests for GeneralFunctions module
"""


def test_upgradedata():
    """ Test the upgradedata function with 2 pandas arrays"""
    x = pd.DataFrame([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9],
                      [10, 11, 12]])
    y = pd.DataFrame([[7, 8, 9, 4, 5, 6, 1, 2, 3],
                      [10, 11, 12, 7, 8, 9, 4, 5, 6]], index=(2, 3))
    assert upgradedata(x).all() == y.all()
