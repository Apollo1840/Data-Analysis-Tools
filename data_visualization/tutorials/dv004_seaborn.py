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



sns.distplot(data_train.SalesPrice)

sns.pairplot(data_train)


sns.set(style='white', context='notebook', palette='deep')
g = sns.countplot(Y_train)



# heatmap
# https://seaborn.pydata.org/generated/seaborn.heatmap.html
import seaborn as sns
ax = sns.heatmap(uniform_data)