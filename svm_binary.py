#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 23:48:32 2017

@author: farhan
"""

import numpy as np
import pandas as pd
data=pd.read_csv("training.csv",header=0,index_col=0)
data=data.replace(np.nan,'',regex=True)
seed=np.loadtxt("development-sets-seeds.gz")
seeds=seed.reshape(seed.shape[1],seed.shape[0])
del seed
data=data.as_matrix()
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score as auc_s
import pickle
#for seed in seeds:
auc_scores=np.zeros(5)
k=0
for seed in seeds:
    X_train=data[seed==0,:-1]
    Y_train=data[seed==0,-1]
    X_test=data[seed==1,:-1]
    Y_test=data[seed==1,-1]
    vect=pickle.loads(dict_vect)
    X_tr0,X_ts0=vect.transform(X_train[:,0]),vect.transform(X_test[:,0])
    X_tr1,X_ts1=vect.transform(X_train[:,1]),vect.transform(X_test[:,1])
    X_tr=np.concatenate((X_tr0.toarray(),X_tr1.toarray()),axis=1)
    X_ts=np.concatenate((X_ts0.toarray(),X_ts1.toarray()),axis=1)
    Y_tr,Y_ts=np.zeros(len(Y_train)), np.zeros(len(Y_test))
    Y_tr[Y_train=="Normal"]=1
    Y_ts[Y_test=="Normal"]=1
    Y_tr[Y_train=="Normal"]=1
    Y_ts[Y_test=="Normal"]=1
    clf=SVC(kernel='rbf',C=1,probability=True)
    scores=cross_val_score(clf,X_tr,Y_tr,cv=5, scoring='f1_macro')
    print(scores)
    clf.fit(X_tr,Y_tr)
    y_pred=clf.predict_proba(X_ts)
    auc_scores[k]=auc_s(Y_ts,y_pred[:,1])
    k+=1
    print(auc_scores)
    