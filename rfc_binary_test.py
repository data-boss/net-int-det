#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 23:15:19 2017

@author: farhan
"""

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
from sklearn.metrics import roc_curve
from itertools import cycle
data=pd.read_csv("clean_balanced_data.csv",header=None,index_col=0,sep=",")
data=data.replace(np.nan,'',regex=True)
#seed=np.loadtxt("development-sets-seeds.gz")
#seeds=seed.reshape(seed.shape[1],seed.shape[0])
#del seed
data=data.as_matrix()
#data=data[data[:,:-1]!='',:]
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score as auc_s
from matplotlib import pyplot as plt
import pickle
#for seed in seeds:
auc_scores=np.zeros(5)
scores=np.zeros((1,5))
from sklearn.model_selection import train_test_split
fpr=dict()
tpr=dict()
roc_auc=dict()
for k in range(1):
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
    Y_tr[Y_train=="Normal"]=1
    Y_ts[Y_test=="Normal"]=1
    clf=RFC(class_weight='balanced',n_estimators=20,max_depth=3,max_features=None)
    scores[k,:]=cross_val_score(clf,X_tr,Y_tr,cv=5, scoring='roc_auc')
#    print(scores)
    clf.fit(X_tr,Y_tr)
    y_pred=clf.predict_proba(X_ts)
    roc_auc[k]=auc_s(Y_ts,y_pred[:,1])
    fpr[k],tpr[k],_=roc_curve(Y_ts,y_pred[:,1])
#    auc_scores[k]=auc_s(Y_ts,y_pred[:,1])
#    
#    k+=1
linestyle = cycle(['dashed', 'solid'])
marker = cycle(["*","o","s","^"])
for i,  marker in zip(range(1), marker):
    plt.plot(fpr[i], tpr[i], lw=1.5, 
             label='Classifier type = "Random Forests" (area = {1:0.2f})'
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
print(time.time()-start_time)