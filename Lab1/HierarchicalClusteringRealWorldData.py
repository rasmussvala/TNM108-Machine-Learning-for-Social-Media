import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.cluster import AgglomerativeClustering

# 200x5 matris
# Innehåller CustomerID, Genre, Age, Annual Income, & Spending Score
customer_data = pd.read_csv('shopping_data.csv')

data = customer_data.iloc[:, 3:5].values # Tar bort kategorierna CustomerID, Genre & Age 

#plt.figure(figsize=(10, 7)) # Storlek på plotten
#plt.subplots_adjust(bottom=0.1) # Plottar dendrogrammet längst ner
#plt.scatter(data[:, 0], data[:, 1], label="True Position") # Plottar punkterna

#plt.show() 

# --------- ny plott ---------

linked = linkage(data, method="average")
# methods: single, ward, average 
plt.figure(figsize=(10, 7))
dendrogram(
    linked,
    orientation="top", # Plottar dendrogrammet uppifrån
    distance_sort="descending", # Sorterar klustren efter avstånd
    show_leaf_counts=True, # Visar antalet datapunkter i varje kluster
)

plt.show()

# --------- ny plott ---------

cluster = AgglomerativeClustering(n_clusters=7, metric='euclidean', linkage='ward')
cluster.fit_predict(data)
plt.scatter(data[:,0],data[:,1], c=cluster.labels_, cmap='rainbow')
plt.show() 