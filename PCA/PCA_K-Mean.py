# -*- coding: utf-8 -*-
"""
Created on Tue May  5 01:35:43 2020

@author: oyedeepak
"""

#importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset 
df= pd.read_csv(r"C:\Users\oyedeepak\Downloads\Assignment\PCA\wine.csv")


#normalization

def norm_func(i):
    x= (i-i.min())/(i.max()-i.min())
    return (x)

# normalizing the dataframe
df_norm= norm_func(df.iloc[:,1:])


#copying the data for hierarchical clustering
df_h= df_norm.copy()


df_h.describe()

### Performing Hierarchical clustering on original data without PCA ###


from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z= linkage(df_h, method= 'ward', metric= 'euclidean')

plt.figure(figsize=(15,5))
plt.title('Hierarchical Clustering')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(z, leaf_rotation=0., leaf_font_size= 8.)
plt.show()

#from Dendrogram, we can take n=3

from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters= 3, linkage= 'ward', affinity= 'euclidean').fit(df_h)

clust_labels= pd.Series(hc.labels_)



### Performing K-Means clustering on original data without PCA ###

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

#copying the normalized data to df_k
df_k= df_norm.copy()

df_k.describe()


#scree plot or elbow curve
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df_k)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#from elbow curve we can take n=3
kmeans= KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=5, random_state=0)

y_pred= kmeans.fit_predict(df_k)



#applying PCA to the dataset

from sklearn.decomposition import PCA
pca= PCA(n_components= 3)
df_pca= pca.fit_transform(df_norm)
explained_variance= pca.explained_variance_ratio_

#cumulative variance
var= np.cumsum(np.round(explained_variance, decimals=4)*100)

### Performing Hierarchical clustering on original data with PCA ###

df_pca_h= df_pca.copy()

z= linkage(df_pca_h, method= 'ward', metric= 'euclidean')

plt.figure(figsize=(15,5))
plt.title('Hierarchical Clustering')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(z, leaf_rotation=0., leaf_font_size= 8.)
plt.show()

#from the Dendrogram, we can say there are 3 clusters, thus n=3

pca_hc= AgglomerativeClustering(n_clusters= 3, linkage= 'ward', affinity= 'euclidean').fit(df_pca_h)

clust_labels_pca= pd.Series(pca_hc.labels_)




### Performing K-Means clustering on original data without PCA ###


#copying the normalized data to df_pca_k
df_pca_k= df_norm.copy()

df_pca_k.describe()


#scree plot or elbow curve
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df_pca_k)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#from elbow curve we can take n=3


#from elbow curve we can take n=3
kmeans= KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=5, random_state=0)

y_pred_pca= kmeans.fit_predict(df_pca_k)
