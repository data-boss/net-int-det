#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 22:34:41 2017

@author: farhan
"""


import numpy as np

data=np.load("june14-seq-features_1.npy")
n=len(data)
p=np.random.permutation(n)
data=data[p,:]
features=data[:,:-1]
labels=data[:,-1]
labels[labels=="Normal"]=0
labels[labels=="Attack"]=1
from sklearn.cross_validation import train_test_split
X,X_ts,y,y_ts=train_test_split(features,labels)



from sklearn.ensemble import AdaBoostClassifier as adb
from sklearn.metrics import roc_auc_score as auc
no_est=[20,100,200,500]
max_feat = [20,50, int(0.5*X.shape[1])]
for j in range(len(no_est)):
    clf=adb(n_estimators=no_est[j])
    clf.fit(X,list(y))
    y_pred=clf.predict_proba(X_ts)
    auc_score=auc(list(y_ts),y_pred[:,1])
    print(auc_score)