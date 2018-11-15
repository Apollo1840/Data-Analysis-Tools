# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 22:39:38 2018

@author: zouco
"""

import pandas as pd 
import numpy as np
from pandas import Series,DataFrame


import os
data_train = pd.read_csv(os.path.dirname(__file__)+'\\datasets\\k000_titanic\\train.csv')
data_train = pd.read_csv('datasets\\k000_titanic\\train.csv')
data_test = pd.read_csv('datasets\\k000_titanic\\test.csv')

# ---------------------------------------------------------------------------
# know the data type and missing data
data_train.info()


# taste analysis
taste_analysis = False
if taste_analysis == True:
    import matplotlib.pyplot as plt
    
    plt.title('dead/survive(0/1)')
    data_train.Survived.value_counts().plot(kind='bar')
    plt.ylabel('people')  
    plt.show()
    
    plt.title('pclass distribution')
    data_train.Pclass.value_counts().plot(kind="bar")
    plt.ylabel('people')
    plt.show()
    
    plt.title('Age and survival ratio')
    plt.scatter(data_train.Survived, data_train.Age,alpha=0.05)
    plt.ylabel("Age")                        
    plt.grid(axis='y',b=True, which='major') 
    plt.show()
    
    
    plt.title('Age and pclass')
    data_train.Age[data_train.Pclass == 1].plot(kind='kde')   
    data_train.Age[data_train.Pclass == 2].plot(kind='kde')
    data_train.Age[data_train.Pclass == 3].plot(kind='kde')
    plt.xlabel("Age")# plots an axis lable
    plt.ylabel("density") 
    plt.legend(('1', '2','3'),loc='best') # sets our legend for our graph.
    
    
    plt.title("embark location")
    data_train.Embarked.value_counts().plot(kind='bar')
    plt.ylabel("people")
    plt.show()


    # primary analysis
    data_train.groupby('Pclass')['Survived'].agg(np.mean).plot('bar')
    
    data_train.groupby('Sex')['Survived'].agg(np.mean).plot('bar')
    
    data_train.groupby('Embarked')['Survived'].agg(np.mean).plot('bar')
    
    data_train.groupby('SibSp')['Survived'].agg(np.mean).plot('bar')
    
    data_train.groupby(['Pclass','Sex'])['Survived'].agg(np.mean).plot('bar')


    Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
    Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
    df=pd.DataFrame({'not null':Survived_cabin, 'null':Survived_nocabin}).transpose()
    print(df)
    df.plot(kind='bar', stacked=True)


# ---------------------------------------------------------------------------
# fill the age
def fill_null_age_rfr(df):
    columns = ['Age','Fare', 'Parch', 'SibSp', 'Pclass']
    known_age = df.loc[df.Age.notnull(), columns].values
    
    from sklearn.ensemble import RandomForestRegressor
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(known_age[:, 1:], known_age[:, 0])
    
    return rfr
    
def fill_null_age(df, rfr):
    columns = ['Age','Fare', 'Parch', 'SibSp', 'Pclass']
    unknown_age = df.loc[df.Age.isnull(), columns].values
    df.loc[ (df.Age.isnull()), 'Age' ] = rfr.predict(unknown_age[:, 1::])
    return df

rfr = fill_null_age_rfr(data_train)

# data preparation
'''
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
lb.fit(list(df.Sex.value_counts().index)).transform(df.Sex)
'''

def data_preprocessing(data_train, rfr):

    if np.sum(data_train.Age.isnull()) != 0:
        data_train = fill_null_age(data_train, rfr)
    data_train.loc[data_train.Fare.isnull(), 'Fare' ] = 0
    
    dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
    dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
    dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')
    
    df = pd.concat([data_train, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    # df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Embarked'], axis=1, inplace=True)
    df = df.filter(regex='Survived|Age|SibSp|Parch|Fare|Cabin|Embarked_.*|Sex_.*|Pclass_.*')
    
    # Cabin
    df.loc[df.Cabin.notnull(), 'Cabin' ] = 1
    df.loc[df.Cabin.isnull(), 'Cabin' ] = 0
    
    # Scale the value : Age, Fare
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    df['Age_scaled'] = scaler.fit_transform(df.loc[:,'Age'].values.reshape(-1,1))
    df['Fare_scaled'] = scaler.fit_transform(df.loc[:,'Fare'].values.reshape(-1,1))
    
    df.drop(['Age', 'Fare'], axis=1, inplace=True)

    print(df.columns)
    
    return df


df = data_preprocessing(data_train, rfr)



# ---------------------------------------------------------------------------
# logistic model:
x_train = df.iloc[:,1:].values
y_train = df.iloc[:,0].values  

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=2, penalty='l1', tol=1e-8)
lr.fit(x_train,y_train)

# evaluate the model
print(lr.score(x_train,y_train))

from sklearn.model_selection import cross_val_score
print(cross_val_score(lr, x_train, y_train, cv=5))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(lr.predict(x_train), y_train))


# predict
df_test = data_preprocessing(data_test, rfr)
x_test = df_test.values

predictions = lr.predict(x_test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result.to_csv("datasets\\k000_titanic\\logistic_regression_predictions.csv", index=False)


# improve:
pd.DataFrame({"columns":list(df.columns)[1:], "coef":list(lr.coef_.T)})



from sklearn.ensemble import BaggingRegressor

lr = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
bagging_clf = BaggingRegressor(lr, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(x_train, y_train)

predictions = bagging_clf.predict(x_test)

result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result.to_csv("datasets\\k000_titanic\\logistic_regression_predictions2.csv", index=False)












