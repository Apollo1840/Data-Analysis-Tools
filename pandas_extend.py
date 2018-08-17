# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 15:10:42 2018

@author: zouco
"""
import pandas as pd
import numpy as np

def choose_columns(df, dt_type='float64'):
    # pick all the column names which contains dt_type data
    
    # use case:
    # float_c = choose_columns(df, 'float64')
    # attr = random.sample(float_c, 2)
    
    columns = []
    for column_name in df.columns:
        if df[column_name].dtype == dt_type:
             columns.append(column_name)
    return columns

def quantile_cut_column(df, column_name, quantile_ratio=[0,0.2,0.4,0.6,0.8,1], labels = ['small','medium-small','medium','medium-large','large']):
    # cut the column by quantile, returns the new df (with a new column  '..._level')
    
    # use case:
    # df2=quantile_cut_column(df, 'title_year')
    # print(df2)
    
    bins = list(df[column_name].quantile(quantile_ratio))
    print(bins)
    bins[0]=bins[0]-0.1
    df[column_name + '_level']=list(pd.cut(df[column_name], bins=bins, labels=labels))
    return df


class da_opt():
    # import da_opt as dp
    # use case:  dp.mean()
    
    trust_amount = 10
    replace_num = np.nan    
    
    @classmethod
    def mean(cls, x):
        if len(x) >= cls.trust_amount:
            return np.mean(x)
        else:
            return cls.replace_num

def df_to_hm_data(df):
    # df to heatmap data, enumerate the df by row and column
    
    value = []
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            value.append([i,j,df.iloc[i,j]])
    return value




def pivot_table_df(df, index, column, value, aggfunc):
    # columns can only be str. aggfunc can only have one func
        df2 = pd.pivot_table(df, index = index, columns = column, values = value, aggfunc = [aggfunc])
        df2 = pd.DataFrame(df2.to_records())
        df2.columns = [hdr.replace("('"+aggfunc.__name__+"', ", column+".").replace(")", "") for hdr in df2.columns]
        return df2 


def sortDF_column(df, column_order):
    df2 = df.T
    df2['columns'] =  df2.index
    df2.index = column_order
    df2 = df2.sort_index()
    columns = df2['columns']
    del df2['columns']
    df3 = df2.T
    df3.columns = columns
    return df3

def r_table(a, table_type='freq'):
    temp={}
    for i in a:
        if i in temp:
            temp[i]+=1
        else:
            temp.update({i:1})
    if table_type == 'dict':
        return temp
    else:
        df=pd.DataFrame()
        df['level']=temp.keys()
        df['freq']=temp.values()
        return df.sort_values(by='freq', ascending=False)        


def drop_outliers(df, list_column, level = 4):
    for i in list_column:
        m=np.nanmean(df[i])
        segma = np.std([j for j in df[i] if j >= df[i].quantile(0.1) and j <= df[i].quantile(0.9)])
        outlier_index = [np.abs(j-m)/segma for j in df[i]]
        df = df.drop(df.index[list(np.where(np.array(outlier_index) >= level))])
    return df

class multi_content_column():
    
    def __init__(self,column,sep=' '):
        self.column=column
        self.sep=sep
    
    @property
    def value(self):
        # break the content into several elements in list
        value = []
        for i in self.column:
            value.append(i.split(self.sep))
        return value
    
    @property
    def list_levels(self):
        return [len(i) for i in self.value]
    
    @property
    def classes(self):
        content = []
        for i in self.column:
            content.extend(i.split(self.sep))
        return r_table(content, 'dict')
    
    @property
    def num_classes(self):
        return len(self.classes.keys())
    
    def list_classes(self, order='alpha', lb=0):
        if order=='alpha':
            return sorted(self.classes.keys())
        elif order=='freq':
            return [(i,j) for j,i in sorted(zip(self.classes.values(), self.classes.keys()), reverse=True) if j>=lb]
        else:
            return self.classes.keys()
        
    def contains(self, content):
        # returns a list of bool, length equal to number of rows, True means the content is in that row.
        return [content in i for i in self.value]
    
    
        
        

if __name__=='__main__':
    mcc=multi_content_column(df.genres,'|')
    mcc2=multi_content_column(df.plot_keywords,'|')
    print(mcc.num_classes)
    # print(mcc.list_of_classes('freq'))
    # print(mcc2.list_of_classes('freq',lb=10))
    print(r_table(df.title_year))
    
    a=[[1,2],[3,5],[5]]
    [5 in i for i in a]
    
 
