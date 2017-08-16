#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 00:22:25 2017

@author: farhan
"""

import pandas as pd
import numpy as np

data=pd.read_csv("all_nopayloads_features.csv",index_col=0,header=0)

X=data.iloc[:,:-1]
Y=data.iloc[:,-1]

from sklearn.decomposition import PCA
pca=PCA(n_components=15)
X=pca.fit_transform(X.iloc[:50000,:])

Y=Y.iloc[:50000]
from sklearn.model_selection import train_test_split as tts
X_tr,X_ts,Y_tr,Y_ts=tts(X,Y,test_size=0.2)

from sklearn.svm import OneClassSVM as osvm
clf=osvm(nu=0.01,kernel='linear')
clf.fit(X_tr)
y_pred=clf.predict(X_ts)
Y_ts=Y_ts.as_matrix()
Y_ts[Y_ts=="Normal"]=1
Y_ts[Y_ts!=1]=-1
n_errors=y_pred[y_pred!=Y_ts].size