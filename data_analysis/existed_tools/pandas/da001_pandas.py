import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
    
    # 0 - create DataFrame
    
    # 0.1 create
    
    # way 1
    data = [[1, 2], [3, 4], [5, 6]] 
    df0 = pd.DataFrame(data, index=[1,2,3], columns=['x','y'])
    print(df0)
    
    # way 2
    data = {'id': [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            'x': [1.2, 2, 3, 4, 5.6, 6, 7, 8, 9.4, 10, 11, 12],
            'y': [13, 9, 8, 4, np.nan, 0, 5, np.nan, 7, 7, 10, 11],
            'obj': [np.random.choice(['a','b','c']) for _ in range(12)]
            }
    df1 = pd.DataFrame(data)
    print(df1)
    
    # 0.2 read
    
    # read dataFrame
    df2 = pd.read_excel('Projects/datasets/sample.xlsx')
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

    
    
    # show dataFrame
    print(df1.describe())
    print(df1.describe(include='all'))
    print(df1.describe(include=['object']))
    print(df1.describe(include=[np.number]))
    df1['x','y'].describe()

    print(df1.info())  # dtypes + exists null or not
    print(df1.dtypes)
    
    
    # plot the dataframe
    df1.plot()  # it plots as row number is x axis, each column is a instance
    df1.plot(kind='bar')  # this always for pivot table.
    df1.x.plot(kind='hist')
    df1.x.plot(kind='box')
    
    
    # very usefull
    print(df1.values)  
    print(df1.shape)
    print(df1.values.shape)
    


    '''
    小结：
        这里有三种初始化一个DataFrame的方法： matrix, dictionary 和 read
        可以用describe, info, dtypes来查看df的最基本信息
        用plot就可以方便的可视化
        values, shape是基本操作
    
    '''



    
    # -------------------------------------------------------------------------
    # 1 - selection

    # 1.1 - row 
    
    # index
    print(df2.index)
    df2.index = [str(i) for i in df2.index]
    df2[2:3]        # works
    df2['2':'4']    # also works
    df_new = df2.set_index('name')
    df_new[:'ddf']   # until the last 'ddf'
    
    df2.reset_index()
       
    print(df2.loc[4])  # return as series
    print(df2.iloc[4])  # original index from 0 to n
    
    # 1.2 - column
    
    print(df1['y'])
    print(df1[['x','y']])
    print(df1.y)
    
    # 1.3 - sub dataFrame
    
    print(df1.loc[5,'x'])
    print(df1.loc[5, ['x', 'y']])
    print(df1.iloc[:3,:2])
        
    df1.loc[5,'sign']='good'
    
    # 1.4 - conditional selection 
    
    df2.loc[[True, False, True, True, False],'sign']
    df2 
    df2.loc[df2.notes > 40,'sign']
    df2
    # simpler: df2[df2.notes > 40] 

    df2.loc[(df2.gender=='m') | (df2.gender=='f'), :]
    df2.loc[df2.gender.isin(['m', 'f']), :]
   
    # df2.query('gender==["m","f"]')
        
    '''
    you can assign new value after the selection
    '''
    
    # select via data type
    df2.select_dtypes(include = [np.number])
    
    
    
    
    '''
    小结：    
        select的时候使用 loc 和 iloc 就好了
        row select 记住 index 的用法,和多种条件式
        column select 记住 select_dtypes
    '''
    
    
    
    
    
    
    
    # -------------------------------------------------------------------------
    # 2 - Update
    # add row, delete row, add column, delete column, alter column name

    # 2.1 - add row 
    
    # add a row with its index
    df2.loc[6]=['dad', 5, 45, 0]
    print(df2)
    
    # append a new row
    df2.append({'name':'adx'}, ignore_index=True)
    
    res = ['wad',3,54,1]
    df2.append(pd.Series(res,index=df2.columns), ignore_index=True)
    
    # add row by dataFrame
    print(pd.concat([df1, df1], axis=0)) # df1 + df1
        
    # 2.2 - delete row
    
    df2.drop([0,1], axis=0)
    # df2.drop(index=[0,1])
    
    # 1) drop NA:
    df1.dropna()
    df1.dropena(how='any') # default
    df1.dropna(how='all')
    df1.dropna(thresh=2)
    df1.dropna(subset=['x', 'y'])  # only focus on some columns, any()
    
    print(df1)  # this do not change
    
    # fillna
    print(df1.fillna(0))
    print(df1.fillna(method='ffill'))
    
    # 2) drop duplicates:
    df2.drop_duplicates()
    df2['pass'].drop_duplicates()
    df1.drop_duplicates(subset=['id', 'x'])
    df2['pass'].drop_duplicates(keep='last')
    
    # the attribute - subset
    df1.dropna(subset=['x', 'y'])
    df1.drop_duplicates(subset=['id', 'x'])
    
    print(df1)  # this do not change
    
    # 2.3 - add column
    
    df2['gender']=['m','f','m',np.nan,'m','f']
    
    # add column by dataFrame
    print(pd.concat([df1, df1], axis=1))  
    
    # 2.4 - delete column   

    df2.drop(['code'], axis=1)
    #df2.drop(columns = ['code']) 
    
    del df2['code']
    
    df1.dropna(axis='columns')  # drop the columns which contains NA
    
    
        
    '''
    小结：
        添加和删除行列就记住 concat 和 drop 搭配 axis
        关于添加行 还有一个 append, 可以处理 dictionary 和 series
        关于 NA 和 duplicates >
            drop_...(subset=)
            drop_na: how, thresh, fillna(method=)
            drop_duplicates: keep
    
    '''
    
    

    # 2.5 - column name
    
    print(df2.columns)
    print(df2.columns[1])
    
    # rename
    df2.columns = ['name','code','note','pass']
    df3 = df2.rename(columns={'note': 'notes'})
    print(df3)
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    # -------------------------------------------------------------------------
    # 3 - operator of column
    
    # 3.1 - trial operations
    
    df2['new'] = df2.xx + df2.yy
    df2['new'] = df2.xx * df2.yy  # be careful, this is entrywise multiplication by index

        
    # 3.2 - apply function
    
    df2['name']=df2['name'].apply(str.strip)  # remove the first and last blank character
    
    df2.note = df2.note.apply(lambda x: 10*x)
    print(df2)
    
    # 3.3 - map function
    
    df2.name = df2.name.map({'haha':'hah'})
    
    # 3.4 - we have some embeded functions:
    
    # general
    print(df2.notes.describe())
    print(df2.notes.describe().round(2))

    df2.note = df2.note.astype('float')
    df2['pass'] = df2['pass'].astype('int')  # change boolean to 1,0
    df2.land = df2.land.astype('category')  # change object column to category
    df2.land.cat.codes # this returns the category code of the column, everything based on this column will be faster
    
    
    '''
    important !!! :
    
    you can make the category ordered by:
        labels_quality = ['bad','middle','good']
        df.level = pd.cut(df.quality, bins = 3, labels=labels_quality)
        df.level.astype('category', categories=labels_quality, ordered=True)
    
    then:
        df.sort_values('level')
        df.loc[df.level > 'bad']
    
    
    '''
    
    
    # for str
    print(df2.name.str.strip())
    print(df2.name.str.contains('a'))
    print(df2.name.str.replace('haha','hahaha'))
    
    # for value
    print(df2['pass'].sum())
    print(df2['pass'].mean())
    print(df2['pass'].std())
    
    # for object
    print(df2.land.unique())   
    print(df2.land.nunique()) 
    print(df2.age.value_counts())
    print(df2.age.value_counts(dropna=False))
    print(df2.age.value_counts(normalize=True))
    
    







 

    # -------------------------------------------------------------------------
    # 4 - advance
    
    # 4.1 sort    
    print(df2.sort_values(by=['pass', 'notes'])) # return type, logically first pass then notes
    print(df2)  # this do not change
    
    print(df2.sort_values(by=['pass', 'notes'], ascending=False, na_position='first')) # return type, logically first pass then notes
    
    df2.sort_index()
    
    
    # 4.2 merge
    dfl = pd.DataFrame({
            'city': ['beijing ','tokyo','newyork', 'munich'],
            'temp': [23,24,28, 26]
            })
    
    dfr = pd.DataFrame({
            'city': ['beijing ','tokyo','newyork', 'san'],
            'humid': [12,23,14, 19]
            })
    
    dfc = pd.DataFrame({
            'city': ['beijing ','tokyo','newyork','tianjing'],
            'temp': [23,24,28,21],
            'humid': [12,23,14,10]
            })
    
    pd.merge(dfl, dfr, on='city')
    # pd.merge(dfl, dfr, left_on='city', right_on='city')
    pd.merge(dfl, dfr, on='city', how='outer')
    pd.merge(dfl, dfr, on='city', how='left')
    
    pd.merge(dfl, dfc, on='city', how='outer')
    
    # merge 2 tables with same columns will cause duplicate columns
    def merge_without_duplicate(df1,df2,on,how):
        return pd.merge(df1,df2[list(df2.columns.difference(df1.columns)) + [on]], on=on, how=how)
    merge_without_duplicate(dfl,dfc,'city', how='outer')
    
    
    # 4.3 group by
    df2.groupby('pass').count()  # how many non NA values
    df2.groupby('pass')['name'].count()
    df2.groupby(['pass','gender'])['name'].count()  # pivot table

    df2.groupby('pass')['sign'].agg([len])      # nan is included
    df2.groupby('pass')['notes'].agg([len, 'count', np.sum, np.mean])
    df2.groupby('pass')['notes', 'code'].agg([len, np.sum, np.mean])
    
    '''
     pandas has its default 'Grouper' like 'Every Month' especially for date.
     like: 
         df.groupby(Grouper(key='date', freq='60s'))   
         df.groupby(['name', pd.Grouper(key='date', freq='M')])['ext price'].sum()
         df.groupby(['name', pd.Grouper(key='date', freq='A-DEC')])['ext price'].sum()  # end of a year
         
         more info: http://pandas.pydata.org/pandas-docs/stable/timeseries.html
    '''
    
    # make the result to be normal dataframe for further development
    df2_2 = df2.groupby('pass')['notes'].agg([len, np.sum, np.mean])
    print(df2_2)
    
    df2_2 = df2_2.reset_index()  
    print(df2_2)
    
    # pivot table
    df2=pd.read_excel('datasets/PUBG.xlsx')
    df2
    
    df3=pd.pivot_table(df2, index = 'Level', columns = 'Gender', values = 'Kills', aggfunc = [np.mean])
    print(df3)
    
    df3=pd.pivot_table(df2, index = 'Level', columns = 'Gender', values = 'Kills', aggfunc = [np.mean], 
                       margins = True) # margins means entry for all
    print(df3)
    
    # final
    df3=pd.pivot_table(df2, 
                       index = ['Level', 'Group'], 
                       columns = ['Gender','Place'], 
                       values = 'Kills', 
                       aggfunc = [np.mean],
                       margins = True)
    print(df3)
    
   
    
    
    
    
    
    
    
    # ------------------------------------------------------------------------
    # 5 - trick
    
    # 5.1 transform the column
    df2.name.isnull()  # return a new boolean column, in which NA is False
    df2.isna().any(axis=1)
    df2.name.duplicated(keep=None)  # default is keep = 'first'
    
    
    
    x=np.array([1,2,3,4,5])
    print(x>2)
    print(np.where(x>2))  # return index of true
    print(np.where(x>2,'big','small'))  # return list
    # print(np.where(x>2, [1,2,3,4],[0,0,0,0])) # return array
    
    # for assignment, can not use df2.age
    df2['age']=np.where(df2.Kills>2,'big','small')
    df2
    
    

    # 5.2 pd.cut
    print(df1.x)
    pd.cut(df1.x, bins=4)
    pd.cut(df1.x, [0,3,6,9,12], labels=['a','b','c','d'])
    
    
    # 5.3 split the column
    'zou congyu'.split(' ')
    df2['supervisor']=['Feng Shangsu','Zou Congyu','Oh Sehun','S Zuu', 'de dfad','tr saf']
    df_new = pd.DataFrame([name.split(' ') for name in df2.supervisor], columns = ['firstname','lastname'])
    df_new
    pd.concat([df2, df_new],axis=1)
    
    
    # 5.4 crosstab
    pd.crosstab(series1, series2)
    
    
    # 5.5 multiple map
    # for example, in df2, I want a new column, which maps code 1,2 as prior, map code 2,3,4 as nieder
    df2['class'] = df2.code.isin([1,2]).map({True: 'prior', False: 'nieder'})
    print(df2)
    
    # more abstract:
    # (condition).map({True:x, False:y})
    
    
    # 5.6 work with datetime
    # This part is from Markham: https://github.com/justmarkham/pandas-videos/blob/master/pandas.ipynb
    ufo = pd.read_csv('http://bit.ly/uforeports')
    ufo['Time'] = pd.to_datetime(ufo.Time)
    print(ufo.Time.dt.hour)
    print(ufo.Time.dt.weekday_name)
    print(ufo.Time.dt.dayofyear)
    
    ts = pd.to_datetime('1/1/1999')
    print(ufo.loc[ufo.Time >= ts, :].head())
    
    # it works just like datetime object in datetime
    (ufo.Time.max() - ufo.Time.min()).days
    
    # tricky plot
    ufo.Time.dt.year.value_counts().sort_index().plot()
    
    # pad datetime
    ufo['month'] = ufo.Time.dt.month.astype('str').str.pad(width=2,fillchar='0')
    ufo['new_date'] = ufo.Time.dt.year.str.cat(ufo.month, sep='-')
    
    
    
    
    
    
    
    
    
    
    # ------------------------------------------------------------------------
    # 6 - statistics
    df2
    
    # sampling
    df2.sample(2)
    df2.sample(2, weights=[0.1,0.1,0.1,0.1,0.3,0.3])
    df2.sample(2, weights=[0.1,0.1,0.1,0.1,0.3,0.3], replace = True)
    
        
    # train_test_split by pandas
    train = df1.sample(frac=0.7)
    test = df1.loc[~df1.index.isin(train.index),:]
    
    # correlation
    df1.x.cov(df1.y)
    df1.x.corr(df1.y)
    df2.corr()
    
    # dummies
    pd.get_dummies(df2, columns=['pass'])
    pd.get_dummies(df2, columns=['pass'], drop_first=True) # without first column
    
    # df2['pass'] = df2['pass'].astype('object')
    pd.get_dummies(df2['pass'], prefix='ps')
    
    
    
    
    
    
    
    
    # ----------------------------------------------------------------------
    # 7 - further development
    
    # output 
    df2.to_excel('sample_output.xlsx')
    
    # display option
    pd.set_option('display.max_columns', None)
    pd.reset_option('display.max_rows')
    
    pd.describe_option('float')  # search display options for keyword - 'float'
    pd.reset_option('all')
    
    
    # apply can also apply to dataFrame
    df1.apply(np.argmax, axis=1)
    df1.applymap(int)   # apply this function to all cells
    
    # iterration
    for index, row in df.iterrows():
        print(index, row.abc, row.efg)
        
    # long table
    print(df1)
    df_stack = df1.stack()
    df0 = df_stack.reset_index()
    df0.columns=['id', 'category', 'value']
    df0
    
    df_1 = df_stack.unstack()

    
    
    #---------------------------------------------------------------------
    # 8 - bonus
    df1.memory_usage(deep=True)
    

    

    

    
    


    

















