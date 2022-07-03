# -*- coding: utf-8 -*-
"""
Created on Fri May 13 11:43:15 2022

@author: Devashish
"""
# Import necessary libraries 
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Read csv files from kaggle dataset as Pandas Dataframe
# Change the file path as the file path in your computer
train = pd.read_csv(r"D:\CDAC ML\Cases\Kaggle\Bike Sharing Demand\train.csv") #Rememder to change path as in your computer
test = pd.read_csv(r"D:\CDAC ML\Cases\Kaggle\Bike Sharing Demand\test.csv") #Rememder to change path as in your computer

# X is a feature 
X=train.drop(['datetime','casual','registered','count'],axis=1)

# Scaling X (feature) of train and test dataset
scaler=StandardScaler()
X_trn_scaled=scaler.fit_transform(X)
X_tst_scaled=scaler.transform(test.iloc[:,1:])

#y is a label
y=train['count']

########  k nearest neighbor  #####
knn=KNeighborsRegressor(n_neighbors=18)
knn.fit(X_trn_scaled,y)

# Making predictions on test set
y_pred=np.round(knn.predict(X_tst_scaled))

# Cretiing Dataframe as per the required format of kaggle competition
submit=pd.DataFrame({'datetime':test.datetime, 'count':y_pred})

# Converting Dataframe into CSV file
submit.to_csv('submit_kaggle_Bike Sharing Demand.csv',index=False)

# Upload above file "submit_kaggle_Bike Sharing Demand.csv" at "https://www.kaggle.com/competitions/bike-sharing-demand/submit"
# OR
# Use kaggle API "kaggle competitions submit -c bike-sharing-demand -f submit_kaggle_Bike Sharing Demand.csv -m "Message_KNN_Algorithm""


