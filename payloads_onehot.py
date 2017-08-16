#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 15:09:14 2017

@author: farhan
"""

import pickle
import numpy as np
with open('payloads_dict.pkl','rb') as input:
    payloads=pickle.load(input)
    labels=pickle.load(input)
    lb=pickle.load(input)
    
    
s_bin=np.empty((1,65))
d_bin=np.empty((1,65))
sl_counter=[]
dl_counter=[]

with open("onehot_payloads.pkl",'wb') as output:
    for c in range(10000):
    
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
    pickle.dump(s_bin,output)
    pickle.dump(d_bin,output)
    pickle.dump(sl_counter,output)
    pickle.dump(dl_counter,output)

del s_bin
del d_bin
del sl_counter
del dl_counter
    
    