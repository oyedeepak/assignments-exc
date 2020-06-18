# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset
df= pd.read_csv(r"C:\Users\oyedeepak\Downloads\Assignment\Neural Network\forestfires.csv")
df.head()

df.loc[df['size_category']== 'small', 'size_category']= 0 #small
df.loc[df['size_category']== 'large', 'size_category']= 1 #large

x= df.iloc[:,2:30].values
y= df.iloc[:,30].values


#splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state= 20)


# feature scaling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train= sc.fit_transform(x_train)
x_test= sc.transform(x_test)


# importing keras and libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

#initilising ANN
classifier= Sequential()

#x.shape
classifier.add(Dense(output_dim= 15, init= 'uniform', activation= 'relu', input_dim= 28))
classifier.add(Dense(output_dim= 15, init= 'uniform', activation= 'relu'))
classifier.add(Dense(output_dim= 15, init= 'uniform', activation= 'relu'))
classifier.add(Dense(output_dim= 15, init= 'uniform', activation= 'relu'))
classifier.add(Dense(output_dim= 1, init= 'uniform', activation= 'sigmoid'))

#compile
classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])

#fitting the ANN to training set
classifier.fit(x_train, y_train, batch_size=10, epochs=50)

#preditcting the test set results
y_pred= classifier.predict(x_test)
y_pred = (y_pred> 0.5)

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)

accuracy_score= (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])
print(accuracy_score)
