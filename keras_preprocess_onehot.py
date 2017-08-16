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
s_bin=np.empty((1,65))
d_bin=np.empty((1,65))
sl_counter=[]
dl_counter=[]
for c in range(50000):
    
    if list(payloads[c,0])!=[]:
        temp1=lb.transform(list(payloads[c,0]))
        sl_counter.append(1)
    else:
        temp1=np.zeros((1,65))
        sl_counter.append(len(payloads[c,0]))
    if list(payloads[c,1])!=[]:
        temp2=lb.transform(list(payloads[c,1]))
        dl_counter.append(1)
    else:
        temp2=np.zeros((1,65))
        dl_counter.append(len(payloads[c,1]))
    
    s_bin=np.concatenate([s_bin,temp1],axis=0)
    d_bin=np.concatenate([d_bin,temp2],axis=0)
    
np.savetxt("source_1.csv",s_bin)
np.savetxt("destination_1.csv",d_bin)
np.savetxt("sl_counter_1.csv",sl_counter)
np.savetxt("dl_counter_1.csv",dl_counter)

#del s_bin
#del d_bin
#del sl_counter
#del dl_counter
#s_bin=np.empty((1,65))
#d_bin=np.empty((1,65))
#sl_counter=[]
#dl_counter=[]
#for c in range(100000,200000):
#    
#    if list(payloads[c,0])!=[]:
#        temp1=lb.transform(list(payloads[c,0]))
#        sl_counter.append(1)
#    else:
#        temp1=np.zeros((1,65))
#        sl_counter.append(len(payloads[c,0]))
#    if list(payloads[c,1])!=[]:
#        temp2=lb.transform(list(payloads[c,1]))
#        dl_counter.append(1)
#    else:
#        temp2=np.zeros((1,65))
#        dl_counter.append(len(payloads[c,1]))
#    
#    s_bin=np.concatenate([s_bin,temp1],axis=0)
#    d_bin=np.concatenate([d_bin,temp2],axis=0)
#    
#np.savetxt("source_2.csv",s_bin)
#np.savetxt("destination_2.csv",d_bin)
#np.savetxt("sl_counter_2.csv",sl_counter)
#np.savetxt("dl_counter_2.csv",dl_counter)
#
#del s_bin
#del d_bin
#del sl_counter
#del dl_counter
#s_bin=np.empty((1,65))
#d_bin=np.empty((1,65))
#sl_counter=[]
#dl_counter=[]
#for c in range(200000,300000):
#    
#    if list(payloads[c,0])!=[]:
#        temp1=lb.transform(list(payloads[c,0]))
#        sl_counter.append(1)
#    else:
#        temp1=np.zeros((1,65))
#        sl_counter.append(len(payloads[c,0]))
#    if list(payloads[c,1])!=[]:
#        temp2=lb.transform(list(payloads[c,1]))
#        dl_counter.append(1)
#    else:
#        temp2=np.zeros((1,65))
#        dl_counter.append(len(payloads[c,1]))
#    
#    s_bin=np.concatenate([s_bin,temp1],axis=0)
#    d_bin=np.concatenate([d_bin,temp2],axis=0)
#    
#np.savetxt("source_3.csv",s_bin)
#np.savetxt("destination_1.csv",d_bin)
#np.savetxt("sl_counter_3.csv",sl_counter)
#np.savetxt("dl_counter_3.csv",dl_counter)
#
#del s_bin
#del d_bin
#del sl_counter
#del dl_counter

