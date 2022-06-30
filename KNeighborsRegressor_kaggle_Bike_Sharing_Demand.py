# -*- coding: utf-8 -*-
"""
Created on Fri May 13 11:43:15 2022

@author: Devashish
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

train = pd.read_csv(r"D:\CDAC ML\Cases\Kaggle\Bike Sharing Demand\train.csv")
test = pd.read_csv(r"D:\CDAC ML\Cases\Kaggle\Bike Sharing Demand\test.csv")


X=train.drop(['datetime','casual','registered','count'],axis=1)

#scaler=StandardScaler()
#X_trn_scaled=scaler.fit_transform(X)
#X_tst_scaled=scaler.transform(test.iloc[:,1:])
y=train['count']

########  KNN  #####
knn=KNeighborsRegressor(n_neighbors=18)
knn.fit(X_trn_scaled,y)
