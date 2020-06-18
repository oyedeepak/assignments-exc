# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 17:14:29 2020

@author: oyedeepak
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


df= pd.read_csv("C:\\Users\\oyedeepak\\Downloads\\Assignment\\Random Forest\\Fraud_check.csv")
df.head()


df.loc[df['Taxable.Income']<= 30000, 'Taxable.Income']= 0 #Risky
df.loc[df['Taxable.Income']> 30000, 'Taxable.Income']= 1 #Good or not risky


from sklearn import preprocessing

le= preprocessing.LabelEncoder()
for i in range(0,6):
    df.iloc[:,i]= le.fit_transform(df.iloc[:,i])
df.head()


colnames= list(df.columns)
colnames


predictors= df.iloc[:, [0,1,3,4,5]]
target= df.iloc[:,2]


print(predictors.columns)
print(target.name)


x= predictors
y= target


rf = RandomForestClassifier(oob_score=True, n_estimators=15, criterion= 'entropy')

rf.fit(x,y)
rf.predict(x)


df['rf_pred'] = rf.predict(x)
cols = ['rf_pred','Taxable.Income']
df[cols].head()


from sklearn.metrics import confusion_matrix
confusion_matrix(df['Taxable.Income'], df['rf_pred'])


print('Accuracy: ', (118+476)/(118+476+6+0)*100)