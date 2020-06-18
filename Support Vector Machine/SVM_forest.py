# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:44:44 2020

@author: oyedeepak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# importing dataset
df= pd.read_csv(r"C:\Users\oyedeepak\Downloads\Assignment\Neural Network\forestfires.csv")
df.head()


classification= {'small':0, 'large':1}
df['size_category']= df['size_category'].map(classification)
df.dtypes


x= df.iloc[:,2:30].values
y= df.iloc[:,30].values


 #feature scaling
from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
sc_y= StandardScaler()
x= sc_x.fit_transform(x)
#y= sc_y.fit_transform(y)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.3, random_state= 15)


#fitting SVC to the dataset
from sklearn.svm import SVC
regressor= SVC(kernel= 'linear',  C=5)
regressor.fit(x_train, y_train)
y_pred= regressor.predict(x_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
#accuracy = 94.8 percent