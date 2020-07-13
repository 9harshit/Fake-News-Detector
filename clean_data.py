#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 15:37:21 2020

@author: harshit
"""


import pandas as pd

data = pd.read_csv('dataset_clean.csv')
data = data.dropna()
X = data["text"]
y = data['Label']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=45)
X_train.to_csv('x_train.csv', index = False)
y_train.to_csv('y_train.csv', index = False)

X_test.to_csv('x_test.csv', index = False)
y_test.to_csv('y_test.csv', index = False)

