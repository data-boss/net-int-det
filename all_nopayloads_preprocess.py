#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 23:34:33 2017

@author: farhan
"""
import numpy as np
import pandas as pd
data=pd.read_csv("all_connection_labeled_csv.csv",header=0)
num_data=data[['totalSourceBytes','totalDestinationBytes','totalSourcePackets','totalDestinationPackets','startDateTime','stopDateTime']]
time_start=pd.to_datetime(num_data['startDateTime'])
time_stop=pd.to_datetime(num_data['stopDateTime'])
duration=pd.to_timedelta(time_stop-time_start).dt.seconds
num_data['duration']=duration
num_data=num_data[['totalSourceBytes','totalDestinationBytes','totalSourcePackets','totalDestinationPackets','duration']]
cat_data=data[['appName','direction','sourceTCPFlagsDescription','destinationTCPFlagsDescription','protocolName',]]
payloads=data[['sourcePayloadAsBase64','destinationPayloadAsBase64']]
payloads=payloads.fillna('')
cat_data=cat_data.fillna('')
from sklearn.preprocessing import OneHotEncoder as enc
from sklearn.preprocessing import LabelEncoder as lbe
enc1=lbe()
enc2=enc()
cat_int=np.zeros((cat_data.shape[0],cat_data.shape[1]))
for k in range(cat_data.shape[1]):
    cat_int[:,k]=enc1.fit_transform(cat_data.iloc[:,k])
    
cat_enc=enc2.fit_transform(cat_int)
cat_enc=pd.DataFrame(cat_enc.toarray())
label=data[['Tag']]
pd.concat([num_data,cat_enc,label],axis=1).to_csv("all_nopayloads_features.csv")
