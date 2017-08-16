#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 14:34:16 2017

@author: farhan
"""
import numpy as np
import pandas as pd
data=pd.read_csv("all_connection_labeled_csv.csv",header=0,index_col=0)

payloads=data.iloc[:,5:7]
payloads=payloads.fillna('').as_matrix()
payloads=payloads[payloads.any(1)!='']

labels=data.iloc[:,-1]
labels=labels[payloads.any(1)!='']
labels[labels=="Normal"]=1
labels[labels!=1]=-1
del data
from sklearn.feature_extraction.text import CountVectorizer as CV

vect=CV(analyzer='char',lowercase=False)
vect.fit(payloads[:,0])
classes = vect.vocabulary_.keys()

from sklearn.preprocessing import LabelBinarizer as lbe
lb=lbe()
lb.classes_=np.array(list(classes))

 with open('payloads_dict.pkl','wb') as output:
    pickle.dump(payloads,output)
    pickle.dump(labels,output)
    pickle.dump(lb,output)
