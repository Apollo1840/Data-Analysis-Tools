# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 12:11:29 2018

@author: zouco
"""
import numpy as np
import pandas as pd


data = {'id': [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        'x': [1.2, 2, 3, 4, 5.6, 6, 7, 8, 9.4, 10, 11, 12],
        'y': [13, 9, 8, 4, np.nan, 0, 5, np.nan, 7, 7, 10, 11],
        'obj': [np.random.choice(['a','b','c']) for _ in range(12)]
        }

df1 = pd.DataFrame(data)

df2 = pd.read_excel('datasets/sample.xlsx')




