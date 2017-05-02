#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 23:48:32 2017

@author: farhan
"""
import time
start_time=time.time()
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
from sklearn.metrics import classification_report as cre
import pickle
#for seed in seeds:
auc_scores=np.zeros(5)
k=0
y_pred=[]
for seed in seeds:
    X_train=data[seed==0,:-1]
    Y_train=data[seed==0,-1]
    X_test=data[seed==1,:-1]
    Y_test=data[seed==1,-1]
#    del data
    with open('dict_vect.pickle', 'rb') as handle:
        vect = pickle.load(handle)
    X_tr0,X_ts0=vect.transform(X_train[:,0]),vect.transform(X_test[:,0])
    X_tr1,X_ts1=vect.transform(X_train[:,1]),vect.transform(X_test[:,1])
    X_tr=np.concatenate((X_tr0.toarray(),X_tr1.toarray()),axis=1)
    X_ts=np.concatenate((X_ts0.toarray(),X_ts1.toarray()),axis=1)
    clf=SVC(kernel='rbf',C=1,probability=True,decision_function_shape='ovr')
    scores=cross_val_score(clf,X_tr,Y_train,cv=5, scoring='f1_micro')
#    print(scores)
    clf.fit(X_tr,Y_train)
    y_pred.append(clf.predict(X_ts))
#    auc_scores[k]=auc_s(Y_test,y_pred[:,1])
    k+=1
#    print(auc_scores)

print("cross validation f1 scores")
print(scores)
#print("auc test scores")
#print(auc_scores)
print()
print("Time elapsed = ", time.time()-start_time)
print("Classification Report")
print()
print(cre(Y_test,y_pred[4]))