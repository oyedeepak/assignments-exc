# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 01:15:13 2020

@author: oyedeepak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset
df= pd.read_csv(r"C:\Users\oyedeepak\Downloads\Assignment\Neural Network\50_Startups.csv")
df.head()

x= df.iloc[:, :-1].values
y= df.iloc[:, 4].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
x[:,3]=labelencoder.fit_transform(x[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()

#Avoiding the Dummy Variable Trap
x=x[:,1:]

'''
states= pd.get_dummies(x['State'], drop_first= True)
x= x.drop('State', axis= 1)
# concat the dummy variables

x= pd.concat([x, states], axis= 1)

#x=x.values
#y=y.values
'''

#splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state= 20)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train= sc.fit_transform(x_train)
x_test= sc.transform(x_test)


#Defining Root Mean Square Error As our Metric Function 
from keras import backend
def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# importing keras and libraries
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor


#initilising ANN
classifier= Sequential()

x.shape
classifier.add(Dense(30, input_dim=5,init= 'uniform',  activation='relu'))

classifier.add(Dense(25,init= 'uniform',  activation='relu'))
classifier.add(Dense(20,init= 'uniform',  activation='relu'))
classifier.add(Dense(15,init= 'uniform',  activation='relu'))
classifier.add(Dense(15,init= 'uniform',  activation='relu'))
classifier.add(Dense(10,init= 'uniform',  activation='relu'))
classifier.add(Dense(10,init= 'uniform',  activation='relu'))
classifier.add(Dense(10,init= 'uniform',  activation='relu'))
classifier.add(Dense(10,init= 'uniform',  activation='relu'))


#output layer
classifier.add(Dense(output_dim=1,init= 'uniform',  activation='linear'))

#compile
classifier.compile(optimizer= keras.optimizers.Adadelta(), loss= 'mean_squared_error', metrics= [rmse])


#fitting the ANN to training set
classifier.fit(x_train, y_train, batch_size=10, epochs=100)


#preditcting the test set results
y_pred= classifier.predict(x_test)


from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred)) #r2 score = 0.92
