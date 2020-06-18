# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 19:56:32 2020

@author: oyedeepak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing train and test file
train= pd.read_csv(r"C:\Users\oyedeepak\Downloads\Assignment\Support Vector Machine\SalaryData_Train(1).csv")
test= pd.read_csv(r"C:\Users\oyedeepak\Downloads\Assignment\Support Vector Machine\SalaryData_Test(1).csv")

train.loc[train['Salary']== ' <=50K', 'Salary']= 0 ##less than or equal to 50K
train.loc[train['Salary']== ' >50K', 'Salary']= 1 #greater than 50K

test.loc[test['Salary']== ' <=50K', 'Salary']= 0 #less than or equal to 50K
test.loc[test['Salary']== ' >50K', 'Salary']= 1 #greater than 50K

x_train= train.drop(['Salary'], axis=1)
y_train= train.iloc[:,13].values

x_test= test.drop(['Salary'], axis=1)
y_test= test.iloc[:,13].values

x_train_cat= pd.get_dummies(x_train, drop_first=True)
x_test_cat= pd.get_dummies(x_test, drop_first=True)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc_= StandardScaler()
sc= StandardScaler()
x_train_cat= sc.fit_transform(x_train_cat)
x_test_cat= sc.fit_transform(x_test_cat)
#y_train= sc.fit_transform(y_train)
#y_test= sc.fit_transform(y_test)

#fitting SVC to the dataset
from sklearn.svm import SVC
regressor= SVC(kernel= 'linear',  C=2)
regressor.fit(x_train_cat, y_train)
y_pred= regressor.predict(x_test_cat)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
#accuracy= 84 percent
