# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 17:00:57 2020

@author: oyedeepak
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


df= pd.read_csv("C:\\Users\\oyedeepak\\Downloads\\Assignment\\Random Forest\\Company_Data.csv")
df.head()


df.columns
df.loc[df['Sales']<=10, 'Sales']=0
df.loc[df['Sales']>10, 'Sales']=1


from sklearn import preprocessing

le= preprocessing.LabelEncoder()
for i in range(0,11):
    df.iloc[:,i]= le.fit_transform(df.iloc[:,i])
df.head()


colnames= list(df.columns)
colnames


predictors= df.iloc[:, 1:11]
target= df.iloc[:,0]


print(predictors.columns)
print(target.name)


x= predictors
y= target


rf = RandomForestClassifier(oob_score=True, n_estimators=15, criterion= 'entropy')

rf.fit(x,y)
rf.predict(x)


df['rf_pred'] = rf.predict(x)
cols = ['rf_pred','Sales']
df[cols].head()


from sklearn.metrics import confusion_matrix
confusion_matrix(df['Sales'], df['rf_pred'])


print('Accuracy: ', (322+78)/(322+78+3)*100)
