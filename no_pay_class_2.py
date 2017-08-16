#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 01:25:26 2017

@author: farhan
"""
import pandas as pd
import numpy as np
data=pd.read_csv("all_nopayloads_features.csv",index_col=0,header=0)
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score as score
from sklearn.metrics import classification_report as clr

X,Y=data.iloc[:,:-1],data.iloc[:,-1:]
del data
X_tr,X_ts,Y_tr,Y_ts=train_test_split(X,Y,train_size=0.4,test_size=0.2)
del X
del Y
clf=rfc(n_estimators=20,max_depth=2,max_features="auto",class_weight='balanced')
clf.fit(X_tr,Y_tr)

y_pred=clf.predict(X_ts)
print(clr(Y_ts,y_pred))

from sklearn.ensemble import GradientBoostingClassifier as gbc
clf2=gbc(n_estimators=20,max_depth=2)
clf2.fit(X_tr,Y_tr)
y_pred2=clf2.predict(X_ts)
print(clr(Y_ts,y_pred2))