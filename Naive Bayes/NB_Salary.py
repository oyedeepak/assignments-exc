# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 21:15:33 2020

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

#categorical encoding using get_dummie
x_train_cat= pd.get_dummies(x_train, drop_first=True)
x_test_cat= pd.get_dummies(x_test, drop_first=True)


# train a Gaussian Naive Bayes classifier on the training set
from sklearn.naive_bayes import GaussianNB

# instantiate the model
gnb = GaussianNB()


gnb.fit(x_train_cat, y_train)


#predict the results
y_pred = gnb.predict(x_test_cat)

y_pred


#confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm
acc = np.mean(y_pred == y_test)*100

