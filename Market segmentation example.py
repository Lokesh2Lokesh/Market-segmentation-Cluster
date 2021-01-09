## Import the relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

## Load the data
data = pd.read_csv('3.12.+Example.csv')
print(data.head)
print(data.describe())
## Plot the data
plt.scatter(data['Satisfaction'],data['Loyalty'])
plt.xlabel('Satisfaction')
plt.xlabel('Loyalty')
plt.show()
## Select the features
x = data.copy()
## Clustering
kmeans = KMeans(4)
kmeans.fit(x)
## Clustering results
Clusters = kmeans.fit_predict(x)
data_with_clusters = x.copy()
data_with_clusters['clusters'] = Clusters
print(data_with_clusters)
## Plot the data
plt.scatter(data_with_clusters['Satisfaction'],data_with_clusters['Loyalty'],c=data_with_clusters['clusters'],cmap='rainbow')
plt.title(' KMeans Cluster without Standard Scaler Data')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()

## Standardize the variables
from sklearn import preprocessing
X_scaled = preprocessing.scale(x)
print(X_scaled)

## Take advantage of the Elbow method
wcss = []

kmeans.inertia_
# Using Loop function to find the WCSS values
for i in range(1,10):
    kmeans = KMeans(i)
    kmeans.fit(X_scaled)
    wcss_inter = kmeans.inertia_
    wcss.append(wcss_inter)
print(wcss)
# Plot the WCSS
numer_clusters = range(1,10)
plt.plot(numer_clusters,wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel(' Within CLusters Sum of Squares')
plt.show()

# Explore CLusterning solution with Standardize Scaled data

kmeans_new = KMeans(5)
kmeans_new.fit(X_scaled)
cluster_new = x.copy()
cluster_new['clusters'] = kmeans_new.fit_predict(X_scaled)
# Plot the data again with scale data
plt.scatter(cluster_new['Satisfaction'],cluster_new['Loyalty'],c=cluster_new['clusters'],cmap='rainbow')
plt.title(' KMeans Cluster with Standard Scaler Data')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()
