# -*- coding: utf-8 -*-
"""
Created on Fri May 23 21:48:19 2025

@author: Subhajit Chattopadhy
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 23 21:41:39 2025

@author: Subhajit Chattopadhy
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 23 21:32:44 2025

@author: Subhajit Chattopadhy
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 23 21:25:32 2025

@author: Subhajit Chattopadhy
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 23 20:42:35 2025

@author: Subhajit Chattopadhy
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 17 05:58:30 2025

@author: Subhajit Chattopadhy
"""

import tensorflow as tf
import keras as keras


import pandas as pd

import numpy as np

import seaborn as sns

import os as os


os.chdir('E:/Pub/Use of MLMs/Data/Clustered Data/')

os.listdir()

df = pd.read_excel("vol_cluster5_ann.xlsx")

df.columns


df.shape

dr3 = df.drop(df.columns[[0,15,16,17,18,22,23,24,25]],axis=1)

dr3.columns






predictors = dr3.drop(dr3.columns[[14,15,16]], axis=1)

predictors.columns

y3 = pd.DataFrame({'side' : dr3['ann3_side'], 
                 'up' : dr3['ann3_up'],
                 'down' : dr3['ann3_dn']})




from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(predictors, y3, test_size=0.2)

y_train.shape

y_test.shape


X_train.shape


from keras import Sequential

from sklearn.model_selection import KFold 

from sklearn.metrics import accuracy_score

from keras.layers import Dense

from keras import Input

from keras import Model

    

ann3 = tf.keras.Sequential()
    
ann3.add(keras.Input(shape=(14,)))
    
    
ann3.add(Dense( units = 128, activation = "tanh"))
    
    
ann3.add(Dense(units = 128, activation = "tanh"))
    

ann3.add(Dense(units = 128, activation = "tanh"))

ann3.add(Dense(units = 128, activation = "tanh"))
    

ann3.add(Dense(units = 3, activation = "softmax"))
    
ann3.compile(loss= 'categorical_crossentropy', 
             optimizer = keras.optimizers.RMSprop(learning_rate=0.0001), metrics = ['accuracy']) 


ann3.summary()

fit_train = ann3.fit(X_train, y_train, batch_size =5, epochs = 10, validation_split = 0.2)


np.mean(fit_train.history['accuracy'])



test_acc = ann3.fit(X_test,y_test, batch_size = 5, epochs = 10)

np.mean(test_acc.history["accuracy"])

#############################################################################3



df.columns.shape

dr5 = df.drop(df.columns[[0,15,16,17,18,19,20,21,25]], axis = 1)

dr5.columns


predictors5 = dr5.drop(dr5.columns[[14,15,16]], axis=1)

predictors5.columns

y5 = pd.DataFrame({'ann5_side': dr5['ann5_side'],
                  'ann5_up' : dr5['ann5_up'],
                  'ann5_dn' : dr5['ann5_dn']})



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(predictors5, y5, test_size=0.2)

y_train.shape

y_test.shape


X_train.shape


from keras import Sequential

from sklearn.model_selection import KFold 

from sklearn.metrics import accuracy_score

from keras.layers import Dense

from keras import Input

from keras import Model

    

ann5 = tf.keras.Sequential()
    
ann5.add(keras.Input(shape=(14,)))
    
    
ann5.add(Dense( units = 128, activation = "tanh"))
    
    
ann5.add(Dense(units = 128, activation = "tanh"))
    

ann5.add(Dense(units = 128, activation = "tanh"))

ann5.add(Dense(units = 128, activation = "tanh"))
    

ann5.add(Dense(units = 3, activation = "softmax"))
    
ann5.compile(loss= 'categorical_crossentropy', 
             optimizer = keras.optimizers.RMSprop(learning_rate=0.0001), metrics = ['accuracy']) 


ann5.summary()




fit_train5 = ann5.fit(X_train, y_train, batch_size =5, epochs = 10, validation_split = 0.2)


np.mean(fit_train5.history['accuracy'])



test5_acc = ann5.fit(X_test,y_test, batch_size = 5, epochs = 10)

np.mean(test5_acc.history["accuracy"])
