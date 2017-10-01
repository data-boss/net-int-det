#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 20:54:08 2017

@author: farhan
"""

import pandas as pd

from keras.models import Model
from keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D, GlobalMaxPool1D,AveragePooling1D,GlobalAveragePooling1D
from keras.layers import LSTM, Lambda, Bidirectional, concatenate, BatchNormalization
from keras.layers import TimeDistributed
from keras.optimizers import Adam
import keras.backend as K
import numpy as np
import tensorflow as tf
import re
import keras.callbacks
import sys
import os


def binarize(x, sz=71):
    return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))


def binarize_outshape(in_shape):
    return in_shape[0], in_shape[1], 71


def striphtml(s):
    p = re.compile(r'<.*?>')
    return p.sub('', s)


def clean(s):
    return re.sub(r'[^\x00-\x7f]', r'', s)


total = len(sys.argv)
cmdargs = str(sys.argv)

print ("Script name: %s" % str(sys.argv[0]))
checkpoint = None
if len(sys.argv) == 2:
    if os.path.exists(str(sys.argv[1])):
        print ("Checkpoint : %s" % str(sys.argv[1]))
        checkpoint = str(sys.argv[1])

data = pd.read_csv("payloads_labels.tsv", header=0,index_col=0, delimiter="\t", quoting=3)
data=data.replace(np.nan,'',regex=True)
txt = ''
docs = []
sentences = []
labels = []

for cont1,cont2,lab in zip(data.pay_1, data.pay_2,data.target):
    sentences = [cont1,cont2]
#    sentences = [sent.lower() for sent in sentences]
    docs.append(sentences)
    labels.append(lab)

num_sent = []
for doc in docs:
    num_sent.append(len(doc))
    for s in doc:
        txt += s

chars = set(txt)

print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print('Sample doc{}'.format(docs[1200]))

maxlen = 1368
max_sentences = 2

X = np.ones((len(docs), max_sentences, maxlen), dtype=np.int64) * -1
y = np.array(labels)

for i, doc in enumerate(docs):
    for j, sentence in enumerate(doc):
        if j < max_sentences:
            for t, char in enumerate(sentence[-maxlen:]):
                X[i, j, (maxlen - 1 - t)] = char_indices[char]

print('Sample X:{}'.format(X[1200, 1]))
print('y:{}'.format(y[1200]))

ids = np.arange(len(X))
np.random.shuffle(ids)

# shuffle
X = X[ids]
y = y[ids]

X_train = X[:20000]
X_test = X[30000:40000]

y_train = y[:20000]
y_test = y[30000:40000]



def char_block(in_layer, nb_filter=(64, 100), filter_length=(3, 3), subsample=(2, 1), pool_length=(2, 2)):
    block = in_layer
    for i in range(len(nb_filter)):

        block = Conv1D(filters=nb_filter[i],
                       kernel_size=filter_length[i],
                       padding='valid',
                       activation='tanh',
                       strides=subsample[i])(block)

        # block = BatchNormalization()(block)
        # block = Dropout(0.1)(block)
        if pool_length[i]:
            block = MaxPooling1D(pool_size=pool_length[i])(block)

    # block = Lambda(max_1d, output_shape=(nb_filter[-1],))(block)
    block = GlobalAveragePooling1D()(block)
    block = Dense(128, activation='relu')(block)
    return block


max_features = len(chars) + 1
char_embedding = 40

document = Input(shape=(max_sentences, maxlen), dtype='int64')
in_sentence = Input(shape=(maxlen,), dtype='int64')

embedded = Lambda(binarize, output_shape=binarize_outshape)(in_sentence)

block2 = char_block(embedded, (128, 256), filter_length=(5, 5), subsample=(1, 1), pool_length=(2, 2))
block3 = char_block(embedded, (192, 320), filter_length=(7, 5), subsample=(1, 1), pool_length=(2, 2))

sent_encode = concatenate([block2, block3], axis=-1)
# sent_encode = Dropout(0.2)(sent_encode)

encoder = Model(inputs=in_sentence, outputs=sent_encode)
encoder.summary()

encoded = TimeDistributed(encoder)(document)

lstm_h = 92

lstm_layer = LSTM(lstm_h, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, implementation=0)(encoded)
lstm_layer2 = LSTM(lstm_h, return_sequences=False, dropout=0.1, recurrent_dropout=0.1, implementation=0)(lstm_layer)

# output = Dropout(0.2)(bi_lstm)
output = Dense(1, activation='sigmoid')(lstm_layer2)

model = Model(outputs=output, inputs=document)

model.summary()

if checkpoint:
    model.load_weights(checkpoint)

file_name = os.path.basename(sys.argv[0]).split('.')[0]

check_cb = keras.callbacks.ModelCheckpoint('checkpoints/' + file_name + '.{epoch:02d}-{val_loss:.2f}.hdf5',
                                           monitor='val_loss',
                                           verbose=0, save_best_only=True, mode='min')

earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

optimizer = 'rmsprop'
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=100, epochs=1, shuffle=True, callbacks=[check_cb, earlystop_cb])

y_pred=model.predict(X_test)
fpr,tpr,thresh=roc_curve(y_test,y_pred)
auc_score=roc_auc_score(y_test,y_pred)
from matplotlib import pyplot as plt
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)
plt.plot(fpr, tpr, linestyle='-', lw=2, color='b',
         label='cnn-lstm average pooling (AUC = %0.2f)' %(auc_score))
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristics')
plt.legend(loc="lower right")
plt.show()

auc_score=roc_auc_score(y_test,y_pred)