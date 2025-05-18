# -*- coding: utf-8 -*-
"""
Created on Sat May 17 05:58:30 2025

@author: Subhajit Chattopadhy
"""

import tensorflow as tf
import keras as keras
import pandas as pd

import numpy as np

import sys as sys

import os as os

sys.modules('tensorflow')


os.chdir('E:/Pub/Use of MLMs/Data')

os.listdir()

df = pd.read_csv('E:/Pub/Use of MLMs/Data/final_processed_data.csv')

df.columns

dr3 = df.drop(df.columns[[0,1,3,4,5,10,11,18,22,23,25,26,27,28,29,30]],axis=1)

dr3.columns


dr3.columns.get_loc('dr3')

predictors = dr3.drop(dr3.columns[[14]], axis=1)

predictors.columns

y = dr3.iloc[:,14]

y.unique()

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
 

X_train, X_test, y_train, y_test = train_test_split(predictors, y, test_size=0.2, random_state=42)

 
from keras import Sequential

from keras.layers import Dense

ann3 = tf.keras.Sequential()

ann3.add(Dense(units = 64, activation = "relu", input_dim = X_train.shape[1]))

ann3.add(Dense(units=32, activation = "relu"))

ann3.add(Dense(units = 16, activation = "relu"))

ann3.add(Dense(units = 8, activation = "relu"))

ann3.add(Dense(units = 3, activation = "softmax"))

ann3.summary()

ann3.compile(loss="categorical_crossentropy", optimizer =  "rmsprop", metrics = [ "accuracy"])

ann3_train_fit = ann3.fit(X_train, y_train, batch_size = 32)

ann3_train_fit.summary()

X_train.shape
X_test.shape

y_test.shape
y_train.shape

test_metrics = ann3.evaluate(X_test, y_test)

y_pred = ann3.predict_classes(X_test)


from sklearn.metrics import confusion_matrix

cf_train = confusion_matrix(y_test, y_pred) 
