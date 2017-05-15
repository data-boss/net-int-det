#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 23:48:32 2017

@author: farhan
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
data=pd.read_csv("clean_balanced_data.csv",header=0,index_col=0)
data=data.replace(np.nan,'',regex=True)

data=data.as_matrix()
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score as auc_s
import pickle
#for seed in seeds:
#auc_scores=np.zeros(2)
#k=0

X_train,X_test,Y_train,Y_test=train_test_split(data[:,:2],data[:,2],test_size=0.2)
with open('dict_vect.pickle', 'rb') as handle:
    vect = pickle.load(handle)
X_tr0,X_ts0=vect.transform(X_train[:,0]),vect.transform(X_test[:,0])
X_tr1,X_ts1=vect.transform(X_train[:,1]),vect.transform(X_test[:,1])
X_tr=np.concatenate((X_tr0.toarray(),X_tr1.toarray()),axis=1)
X_ts=np.concatenate((X_ts0.toarray(),X_ts1.toarray()),axis=1)
Y_tr,Y_ts=np.zeros(len(Y_train)), np.zeros(len(Y_test))
Y_tr[Y_train=="Normal"]=1
Y_ts[Y_test=="Normal"]=1
#    Y_tr[Y_train=="Normal"]=1
#    Y_ts[Y_test=="Normal"]=1
clf=SVC(kernel='linear',C=1,probability=True)
scores=cross_val_score(clf,X_tr,Y_tr,cv=5, scoring='f1_macro')
print(scores)
clf.fit(X_tr,Y_tr)
y_pred=clf.predict_proba(X_ts)
auc_scores=auc_s(Y_ts,y_pred[:,1])
#    k+=1
print(auc_scores)
roc_auc=np.zeros(1)
roc_auc[0]=auc_s(Y_ts,y_pred[:,1])
from sklearn.metrics import roc_curve
fpr,tpr,_=roc_curve(Y_ts,y_pred[:,1])
#    auc_scores[k]=auc_s(Y_ts,y_pred[:,1])
#    
#    k+=1
from itertools import cycle
from matplotlib import pyplot as plt
linestyle = cycle(['dashed', 'solid'])
marker = cycle(["*","o","s","^"])
for i,  marker in zip(range(1), marker):
    plt.plot(fpr[i], tpr[i], lw=1.5, 
             label='Classifier type = "SVM with rbf kernel" (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristics')
plt.legend(loc="lower right",fontsize='small')
plt.show()
print()
