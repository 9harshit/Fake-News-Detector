#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 14:04:56 2020

@author: harshit
"""

max_features = 40000
maxlen = 200
embedding_size = 128

from keras.models import load_model
from keras.models import Model, Input
from keras import layers


models=[]
name = ["bin.hdf5", "cnn_lstm.hdf5", "rnn.hdf5", "cnn_text_10.hdf5"]
for i in range(2):
    modelTemp=load_model(name[i]) # load model
    modelTemp.name= name[i] # change name to be unique
    models.append(modelTemp)
    
def ensembleModels(models, model_input):
    # collect outputs of models in a list
    yModels=[model(model_input) for model in models] 
    # averaging outputs
    yAvg=layers.average(yModels) 
    # build model from same input and avg output
    modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')  
   
    return modelEns



from keras.preprocessing import sequence


max_features = 30000
# cut texts after this number of words (among top max_features most common words)
maxlen = 200
batch_size = 32


import pandas as pd

print('Loading data...')
x_train = pd.read_csv("x_train.csv")
x_test = pd.read_csv("x_test.csv")
y_test = pd.read_csv("y_test.csv")

x_train = x_train['text'].fillna('').tolist()
x_test = x_test['text'].fillna('').tolist()


from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words = 2000)
tokenizer.fit_on_texts(x_train)

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


model_input = Input(shape=models[0].input_shape[1:]) # c*h*w
modelEns = ensembleModels(models, model_input)
modelEns.summary()


modelEns.save("esnmodel.h5")

modelEns=load_model("esnmodel.h5")
modelEns.summary()

y=modelEns.predict(x_test)

import numpy as np

pred = np.zeros(11230)

for i in range(len(y)):
    if y[i]> 0.05:
        pred[i] = 1
    else:
        pred[i] = 0
        
y_test = np.float64(y_test)

ans = 0
for x in range(len(pred)):
    if pred[0] == y_test[0]:
        ans += 1