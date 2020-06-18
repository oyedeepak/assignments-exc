# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 19:16:47 2020

@author: oyedeepak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset
df= pd.read_csv(r"C:\Users\oyedeepak\Downloads\Assignment\Neural Network\concrete.csv")
df.head()

x= df.drop('strength', axis=1).values
y=df['strength'].values

#splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state= 20)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train= sc.fit_transform(x_train)
x_test= sc.transform(x_test)


import math
#Defining Root Mean Square Error As our Metric Function 
from keras import backend
def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


# importing keras and libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

#initilising ANN
classifier= Sequential()

x.shape
classifier.add(Dense(16, input_dim=8, activation='relu'))

classifier.add(Dense(16, activation='relu'))
classifier.add(Dense(16, activation='relu'))
classifier.add(Dense(16, activation='relu'))
classifier.add(Dense(16, activation='relu'))

#output layer
classifier.add(Dense(output_dim=1, activation='linear'))

#compile
classifier.compile(optimizer= 'adam', loss= 'mean_squared_error', metrics= [rmse])


#fitting the ANN to training set
classifier.fit(x_train, y_train, batch_size=10, epochs=100)


#preditcting the test set results
y_pred= classifier.predict(x_test)


from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred)) #r2 scoree= 0.85
