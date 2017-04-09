#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 18:24:05 2017

@author: farhan
"""

from gensim import corpora
import numpy as np
import pandas as pd
import sys
file=sys.argv[1]
file2=sys.argv[2]
df=pd.read_csv(file,header=None)
df=df.replace(np.nan,'',regex=True)
df=df.as_matrix()
df_lab=pd.read_csv(file2,header=None)
df_lab=df_lab.as_matrix().reshape(len(df_lab))
df_normal=df[df_lab=="Normal",:]
df_attack=df[df_lab=="Attack",:]
df_lab=np.concatenate([df_lab[df_lab=="Normal"], df_lab[df_lab=="Attack"]], axis=0)
df_payloads=np.concatenate([df_normal, df_attack],axis=0)
del df
df_payloads=np.concatenate([df_payloads[:10000,:],df_payloads[-10000:,:]],axis=0)
df_lab=np.concatenate([df_lab[:10000],df_lab[-10000:]],axis=0)
del df_normal, df_attack
df_s=df_payloads[:,0]
df_d=df_payloads[:,1]

k=int(sys.argv[3])
n=df_s.shape[0]
import nltk
from nltk.util import ngrams
kgrams=[]
for x in range(len(df_s)):
    token=list(df_s[x])
    kgram=list(ngrams(token,k))
    kgram=[''.join(grams) for grams in kgram]
    kgrams.append(kgram)

#kgrams=[gram for gram in kgrams if len(gram)!=0]

dictionary = corpora.Dictionary(kgrams)
d=len(dictionary)
corpus = [dictionary.doc2bow(gram) for gram in kgrams]
features_s=np.zeros((n,d))
for f in range(n):
    if corpus[f]==[]:
        features_s[f,:]=features_s[f,:]
    else:
        temp=np.asarray(corpus[f])
        idx,val=temp[:,0],temp[:,1]
        features_s[f,idx]=val

kgrams=[]
for x in range(len(df_d)):
    token=list(df_d[x])
    kgram=list(ngrams(token,k))
    kgram=[''.join(grams) for grams in kgram]
    kgrams.append(kgram)

#kgrams=[gram for gram in kgrams if len(gram)!=0]

dictionary = corpora.Dictionary(kgrams)
d=len(dictionary)
corpus = [dictionary.doc2bow(gram) for gram in kgrams]
features_d=np.zeros((n,d))
for f in range(n):
    if corpus[f]==[]:
        features_d[f,:]=features_d[f,:]
    else:
        temp=np.asarray(corpus[f])
        idx,val=temp[:,0],temp[:,1]
        features_d[f,idx]=val

features=np.concatenate([features_s, features_d, df_lab.reshape(n,1)],axis=1)
np.save(str(argv[1])+"seq_"+str(k)+".npy",features)
#kgrams_array=np.array([np.array(xi) for xi in kgrams])
#unique_kgrams=np.array([np.unique(yi) for yi in kgrams_array])

    
    
