# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 00:39:58 2020

@author: oyedeepak
"""

#Naive Bayes

# importing libs
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score



# importing datasets
df= pd.read_csv(r"C:\Users\oyedeepak\Downloads\Assignment\Naive Bayes\sms_raw_NB.csv",encoding = "latin-1")

classification= {'ham':0, 'spam':1}
df['type']= df['type'].map(classification)
df.dtypes

x= df['text']
y= df['type']

#splitting into train and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=5)


# feature scaling
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(x_train)
classifier = MultinomialNB()
classifier.fit(counts,y_train)


counts_test = vectorizer.transform(x_test)
prediction = classifier.predict(counts_test)
print('Accuracy score: {}'.format(accuracy_score(y_test, prediction)))
print('Precision score: {}'.format(precision_score(y_test, prediction)))
print('Recall score: {}'.format(recall_score(y_test, prediction)))
print('F1 score: {}'.format(f1_score(y_test, prediction)))