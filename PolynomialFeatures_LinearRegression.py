# -*- coding: utf-8 -*-
"""
Created on Fri May 13 15:22:51 2022

@author: Devashish
"""
# Import necessary libraries 
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Read csv files from kaggle dataset as Pandas Dataframe
# Change the file path as the file path in your computer
train = pd.read_csv(r"C:\Users\Devashish\Downloads\train.csv",parse_dates=['datetime'])
train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['hour'] = train['datetime'].dt.hour
train['weekday'] = train['datetime'].dt.weekday

test = pd.read_csv(r"C:\Users\Devashish\Downloads\test.csv",parse_dates=['datetime'])
test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['day'] = test['datetime'].dt.day
test['hour'] = test['datetime'].dt.hour
test['weekday'] = test['datetime'].dt.weekday

# X is a feature 
x= train.drop(['datetime','casual', 'registered', 'count'],axis=1)

#y is a label
y_c= train['casual']
y_p= train['registered']
X_test=test.drop(['datetime'],axis=1)

poly = PolynomialFeatures(degree=2)
poly_feat = poly.fit_transform(x)
poly_test = poly.transform(X_test)

le_c = LinearRegression()
le_c.fit(poly_feat,y_c)

# Making predictions on test set
y_pred = np.round(le_c.predict(poly_test))
print(y_pred.min())
y_pred[y_pred<0]=0

le_p = LinearRegression()
le_p.fit(poly_feat,y_p)

y_pred_p = np.round(le_p.predict(poly_test))
print(y_pred_p.min())
y_pred_p[y_pred_p<0]=0

y_count=y_pred+y_pred_p

# Cretiing Dataframe as per the required format of kaggle competition
submit = pd.DataFrame({'datetime':test.datetime,'count':y_count})
# Converting Dataframe into CSV file
submit.to_csv("submit_kaggle_Bike Sharing Demand.csv",index=False)

# Upload above file "submit_kaggle_Bike Sharing Demand.csv" at "https://www.kaggle.com/competitions/bike-sharing-demand/submit"
# OR
# Use kaggle API "kaggle competitions submit -c bike-sharing-demand -f submit_kaggle_Bike Sharing Demand.csv -m "Message_PolynomialFeatures_LinearRegression""



