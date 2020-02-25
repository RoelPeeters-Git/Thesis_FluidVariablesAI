# -*- coding: utf-8 -*-
import unittest
import numpy as np
from UpgradeData import upgrade_data

"""
Created on Mon Dec 23 11:49:06 2019

@author: Peeters Roel

A test class to test the upgrade_data function
"""


class TestUpgradeData(unittest.TestCase):

    def test_upgradedata(self):
        """ Test the upgrade_data function with 2 numpy arrays"""
        x = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9],
                      [10, 11, 12]])
        y = np.array([[7, 8, 9, 4, 5, 6, 1, 2, 3],
                      [10, 11, 12, 7, 8, 9, 4, 5, 6]])
        self.assertEqual(upgrade_data(x).all(), y.all())


if __name__ == '__main__':
    unittest.main()
