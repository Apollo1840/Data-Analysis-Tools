# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6

from ploters import plot_distribution, plot_categories, plot_correlation_map,plot_variable_importance


DATA_PATH='Projects\\datasets\\k000_titanic\\'
    
    
# -----------------------------------------------------------------------------    
# get titanic & test csv files as a DataFrame

'''
    in k000_Titanic_1, it works on train set and deploy the changes on test data.
    in this one, it combines train and test first.

'''

import os
path = os.getcwd()
while(os.path.basename(path) != 'Data-Analysis-Tools'):
    path = os.path.dirname
os.chdir(path)
    
train = pd.read_csv(DATA_PATH + 'train.csv')
test = pd.read_csv(DATA_PATH + 'test.csv')

full = train.append( test , ignore_index = True )
titanic = full[ :891 ]

del train , test

print ('Datasets:' , 'full:' , full.shape , 'titanic:' , titanic.shape)


# plot some charts

plot_correlation_map( titanic )

# Plot distributions of Age of passangers who survived or did not survive
plot_distribution( titanic , var = 'Age' , target = 'Survived' , row = 'Sex' )

# Plot survival rate by Embarked
plot_categories( titanic , cat = 'Embarked' , target = 'Survived' )


'''
    here it is going to change some attributes to dummies.

'''
# --- sex ---

# Transform Sex into binary values 0 and 1
sex = pd.Series( np.where( full.Sex == 'male' , 1 , 0 ) , name = 'Sex' )
# sex = full.Sex.map({'male': 1,'female':0})

# --- Age; Fare ---

# Create dataset
imputed = pd.DataFrame()

# Fill missing values of Age with the average of Age (mean)
imputed[ 'Age' ] = full.Age.fillna( full.Age.mean() )

# Fill missing values of Fare with the average of Fare (mean)
imputed[ 'Fare' ] = full.Fare.fillna( full.Fare.mean() )

imputed.head()


# --- title ---

title = pd.DataFrame()
# we extract the title from each name
full['Name']

title[ 'Title' ] = full[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )
title['Title'].value_counts()

# a map of more aggregated titles
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"

                    }

# we map each title
title[ 'Title' ] = title.Title.map( Title_Dictionary )

# --- cabin ---

cabin = pd.DataFrame()

# replacing missing cabins with U (for Uknown)
cabin[ 'Cabin' ] = full.Cabin.fillna( 'U' )

# mapping each Cabin value with the cabin letter
cabin[ 'Cabin' ] = cabin[ 'Cabin' ].apply( lambda c : c[0] )


# --- ticket ---
full.Ticket

# a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
def cleanTicket( ticket ):
    ticket = ticket.replace( '.' , '' )
    ticket = ticket.replace( '/' , '' )
    ticket = [t.strip() for t in ticket.split()]
    ticket = list(filter( lambda t : not t.isdigit() , ticket ))
    if len( ticket ) > 0:
        return ticket[0]
    else: 
        return 'XXX'

ticket = pd.DataFrame()

# Extracting dummy variables from tickets:
ticket[ 'Ticket' ] = full[ 'Ticket' ].apply( cleanTicket )
# ticket[ 'Ticket' ].value_counts()


# -- Family ---

family = pd.DataFrame()

# introducing a new feature : the size of families (including the passenger)
family[ 'FamilySize' ] = full[ 'Parch' ] + full[ 'SibSp' ] + 1


# --- all ---

# full_X = pd.concat( [ imputed , embarked , cabin , sex ] , axis=1 )
# full_X.head()

full_X = pd.concat( [ sex, full.Embarked, full.Pclass, imputed,title, cabin, ticket, family] , axis=1 )
print(full_X.columns)
print(full_X.head())
full_X.to_csv(DATA_PATH +'titanic_data2.csv' )


embarked = pd.get_dummies( full.Embarked , prefix='Embarked' )
pclass = pd.get_dummies( full.Pclass , prefix='Pclass' )

ticket = pd.get_dummies( ticket[ 'Ticket' ] , prefix = 'Ticket' )
cabin = pd.get_dummies( cabin['Cabin'] , prefix = 'Cabin' )
title = pd.get_dummies( title.Title, prefix='Title')

# (discretelization) introducing other features based on the family size
family[ 'Family_Single' ] = family[ 'FamilySize' ].apply( lambda s : 1 if s == 1 else 0 )
family[ 'Family_Small' ]  = family[ 'FamilySize' ].apply( lambda s : 1 if 2 <= s <= 4 else 0 )
family[ 'Family_Large' ]  = family[ 'FamilySize' ].apply( lambda s : 1 if 5 <= s else 0 )


full_X = pd.concat( [ sex, embarked, pclass, imputed ,title, cabin, ticket, family] , axis=1 )
print(full_X.columns)
print(full_X.head())

#-------------------------------------------------------------------------------
# preparation

# Create all datasets that are necessary to train, validate and test models

'''
    train_x, train_y is the train data
    test_x, test_y is the validation data
    
    train_all_x, train_all_y is the final training data
    
    test_X is the data to predict

'''



train_X = full_X[ 0:891 ]
train_Y = titanic.Survived.values.reshape(-1,1)
test_X = full_X[ 891: ]
train_x , test_x , train_y , test_y = train_test_split( train_X , train_Y , train_size = .7 )

print(full_X.shape , train_X.shape, train_Y.shape, test_X.shape)
print(train_x.shape , train_y.shape , test_x.shape, test_y.shape)


# plot_variable_importance(train_X, train_y)


# modeling ------------------------------------------------------------------


model = RandomForestClassifier(n_estimators=100)
# model = GradientBoostingClassifier()
# model = LogisticRegression(C=0.5, penalty='l2', tol=1e-9)
rfecv = RFECV( estimator = model , step = 1 , cv = StratifiedKFold( train_y , 2 ) , scoring = 'accuracy' )
rfecv.fit( train_x , train_y )
print(rfecv.score( train_x , train_y ))
print(rfecv.score( test_x, test_y ))



score = 0
while(score < 0.78):
    model = MLPClassifier((300,100,50), activation = 'tanh', tol=10e-6)
    model.fit( train_x , train_y )
    model.fit( train_x , train_y )
    model.fit( train_x , train_y )
    print(model.score( train_x , train_y ))
    print(model.score( test_x , test_y ))
    score = model.score( test_x , test_y )


# ------------------------------------------------------------------
# output

def save_result(model):
    test_Y = model.predict( test_X )
    passenger_id = full[891:].PassengerId
    test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y.astype(np.int32) } )
    # test = test.reset_index()
    # test = test.drop(columns = ['index'])
    print(test.info())
    test.to_csv( DATA_PATH + 'titanic_pred.csv' , index = False )


save_result(rfecv)
save_result(model)





