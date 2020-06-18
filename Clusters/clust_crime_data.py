# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:51:24 2020

@author: oyedeepak
"""

import pandas as pd
import matplotlib.pylab as plt 
import numpy as np

crime = pd.read_csv('C:/Users/oyedeepak/Downloads/Assignment/Clustering/crime_data.csv')

crime.rename( columns={'Unnamed: 0':'State'}, inplace=True )

#normalisation function
def norm_func(i):
    x = (i-i.min())/(i.max()/i.min())
    return (x)


x= crime.drop(['State'], axis = 1).values
df_norm = norm_func(x)

### Performing Hierarchical clustering ###

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 

df_norm_k = df_norm.copy()
#scree plot or elbow curve
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df_norm_k)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# from the elbow curve we can say, n= 3

kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=1)
y_pred= kmeans.fit_predict(df_norm_k)


plt.scatter(df_norm_k[y_pred == 0,0], df_norm_k[y_pred == 0,1],  s = 50, c= 'red')
plt.scatter(df_norm_k[y_pred == 1,0], df_norm_k[y_pred == 1,1],  s = 50, c= 'blue')
plt.scatter(df_norm_k[y_pred == 2,0], df_norm_k[y_pred == 2,1],  s = 50, c= 'yellow')
plt.show()




### Performing Hierarchical clustering on original data without PCA ###


from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

df_norm_h= df_norm.copy()


z= linkage(df_norm_h, method= 'ward', metric= 'euclidean')

plt.figure(figsize=(15,5))
plt.title('Hierarchical Clustering')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(z, leaf_rotation=0., leaf_font_size= 8.)
plt.show()

#from Dendrogram, we can infer n=3

from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters= 3, linkage= 'ward', affinity= 'euclidean').fit(df_norm_h)

clust_labels= pd.Series(hc.labels_)
