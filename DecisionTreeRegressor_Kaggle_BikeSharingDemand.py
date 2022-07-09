# -*- coding: utf-8 -*-
"""
Created on Wed May 18 17:32:45 2022

@author: Devashish
"""
# Import necessary libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold,KFold
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

# Change the file path as the file path in your computer
import os
os.chdir("D:\CDAC ML\Cases\Kaggle\Tabular Playground Series - Jan 2022")


# Read csv files from kaggle dataset as Pandas Dataframe

train = pd.read_csv(r"train.csv",parse_dates=['date'],index_col='row_id')
train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.day
train['hour'] = train['date'].dt.hour
train['weekday'] = train['date'].dt.weekday

test = pd.read_csv(r"test.csv",parse_dates=['date'],index_col='row_id')
test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day
test['hour'] = test['date'].dt.hour
test['weekday'] = test['date'].dt.weekday


train1= train.drop(['date','num_sold'],axis=1)

# X is a feature 
X=pd.get_dummies(train1,drop_first=True)

#y is a label
y= train['num_sold']

X_test=test.drop(['date'],axis=1)
X_test=pd.get_dummies(X_test,drop_first=True)


############  Grid Search CV ###########
clf = DecisionTreeRegressor(random_state=2022)
kfold = KFold(n_splits=5,shuffle=True,random_state=2022)

params = {'max_depth':[3,4,None],'min_samples_split':[2,10,20],'min_samples_leaf':[1,5,10]}
gcv1 = GridSearchCV(clf,scoring='r2',cv=kfold,param_grid=params)
gcv1.fit(X,y)
print(gcv1.best_params_)
print(gcv1.best_score_)
print(gcv1.best_estimator_)

best_model1=gcv1.best_estimator_

# Making predictions on test set
y_pred1= np.round(best_model1.predict(X_test))
y_pred1[y_pred1<0]=0


# Cretiing Dataframe as per the required format of kaggle competition
submit = pd.DataFrame({'date':test.date,'count':y_pred1})

# Converting Dataframe into CSV file
submit.to_csv("Tabular Playground Series - Jan 2022_submit_18_05_3nd.csv",index=False)



clf = DecisionTreeRegressor(random_state=2022)
kfold = KFold(n_splits=5,shuffle=True,random_state=2022)

params = {'max_depth':[3,4,None],'min_samples_split':[2,10,20],'min_samples_leaf':[1,5,10]}
gcv2 = GridSearchCV(clf,scoring='r2',cv=kfold,param_grid=params)
gcv2.fit(X,y_p)
print(gcv2.best_params_)
print(gcv2.best_score_)


best_model2=gcv2.best_estimator_

# Making predictions on test set
y_pred2= (np.round(best_model2.predict(X_test))+1)
y_pred2[y_pred2<0]=0



y_count=y_pred1+y_pred2

# Cretiing Dataframe as per the required format of kaggle competition
submit = pd.DataFrame({'date':test.date,'count':y_count})

# Converting Dataframe into CSV file
submit.to_csv("Tabular Playground Series - Jan 2022_submit_18_05_3nd.csv",index=False)



