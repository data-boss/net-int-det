#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:48:57 2017

@author: farhan
"""
import pandas as pd
import numpy as np
train=pd.read_csv("training.csv",index_col=0, header=0)
test=pd.read_csv("test.csv",index_col=0,header=0)
data=pd.concat([train,test],axis=0)
data=data.replace(np.nan,'',regex=True)
data_f=data.loc[(data.iloc[:,:2]!='').any(axis=1),:]
normal=data_f.loc[data_f.iloc[:,2]=="Normal",:]
attack=data_f.loc[data_f.iloc[:,2]!="Normal",:]
from sklearn.utils import shuffle
normal=shuffle(normal,n_samples=len(attack))
clean_balanced_data=shuffle(pd.concat([normal, attack]))
clean_balanced_data.to_csv("clean_balanced_data.csv")