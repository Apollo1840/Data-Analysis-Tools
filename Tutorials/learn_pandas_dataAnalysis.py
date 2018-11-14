import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 1. Get data
    
# read dataFrame
df2 = pd.read_excel('datasets/sample.xlsx')
print(df2)
    
# read a part of csv
df3 = pd.read_csv('...', usecols = ['',''], nrows=10)
    
'''
the most powerful read fuction of pandas is read_table,
you can adjust sep,header,names(column) of it.

header is defaul to be 'infer', it means the first row the data, 
you can adjust it to None, and use names to redefine it.

something even more specific:
    dtype={'beer_servings':float}

'''



# ---------------------------------------------------------------
# clean the data
from learn_pandas_dataframes import df1, df2

# 1) deal with duplicates
df2.drop_duplicates()
df2['pass'].drop_duplicates()
df2['pass'].drop_duplicates(keep='last')

print(df2)  # this do not change

# 2) deal with N/A
# drop rows
df1.dropna()
df1.dropna(how='any') # default
df1.dropna(how='all')
df1.dropna(thresh=2)

print(df1)  # this do not change

# + drop columns
df1.dropna(axis='columns')  # drop the columns which contains NA

# + fillna
print(df1.fillna(0))
print(df1.fillna(method='ffill'))


# the attribute - subset
df1.dropna(subset=['x', 'y'])
df1.drop_duplicates(subset=['id', 'x'])



# rename
df2.columns = ['name','code','note','pass']
df3 = df2.rename(columns={'note': 'notes'})
print(df3)


# ------------------------------------------------------------------
# analysis

# trial 
print(df1.shape)
print(df1.describe())
print(df1.describe(include=[np.number])) # default
print(df1.describe(include=['object']))
print(df1.describe(include='all'))

df1['x','y'].describe()

print(df1.dtypes)
print(df1.info())  # dtypes + exists null or not


# see learn_pandas.py



# ------------------------------------------------------------------ 
# report
# output
df2.to_csv('sample_output.csv')
df2.to_excel('sample_output.xlsx')


# plot the dataframe
df1.plot()  # it plots as row number is x axis, each column is a instance
df1.plot(kind='bar')  # this always for pivot table., each columns is a instance
df1.x.plot(kind='hist')
df1.x.plot(kind='box')















