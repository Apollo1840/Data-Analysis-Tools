# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 08:56:01 2018

@author: zouco
"""

import seaborn as sns
import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(25, 7))
sns.violinplot(x='Sex', y='Age', 
               hue='Survived', data=data_train, 
               split=True,
               palette={0: "r", 1: "g"}
              );
