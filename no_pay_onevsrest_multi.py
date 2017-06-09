#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 01:25:26 2017

@author: farhan
"""
import pandas as pd
import numpy as np
data=pd.read_csv("all_nopayloads_features.csv",index_col=0,header=0)
from sklearn.linear_model import SGDClassifier as sgd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score as score
from sklearn.metrics import classification_report as clr
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
X,Y=data.iloc[:,:-1],data.iloc[:,-1:]
# Binarize the output
from sklearn.preprocessing import LabelEncoder as lbe
enc=lbe()
Y=enc.fit_transform(Y)
Y = label_binarize(Y, classes=[0, 1, 2, 3, 4, 5])
n_classes = Y.shape[1]
del data
X_tr,X_ts,Y_tr,Y_ts=train_test_split(X,Y,train_size=0.3,test_size=0.2)
del X
del Y
clf = OneVsRestClassifier(sgd(class_weight='balanced'))
#clf=rfc(n_estimators=50,max_depth=2,max_features="auto",class_weight='balanced')
clf.fit(X_tr,Y_tr)
y_score = clf.fit(X_tr, Y_tr).decision_function(X_ts)


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_ts[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(Y_ts.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
from scipy import interp
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
classes=['Normal','botnet-ddos','bruteforce','bruteforce-ssh','http-dos','insider']
# Plot all ROC curves
from matplotlib import pyplot as plt
from itertools import cycle
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro ({0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro ({0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, linewidth=1,
             label=str(classes[i])+' ({1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('One vs all ROC -- stochastic gradient descent classifiers')
plt.legend(loc="lower right")
plt.show()
