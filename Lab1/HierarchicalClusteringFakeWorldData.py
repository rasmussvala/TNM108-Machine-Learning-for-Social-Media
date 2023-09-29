import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.cluster import AgglomerativeClustering

# Va kreativ med att leka runt med detta
X = np.array(
    [
        [5, 3],
        [10, 15],
        [15, 12],
        [24, 10],
        [30, 30],
        [85, 70],
        [71, 80],
        [60, 78],
        [70, 55],
        [80, 91],
    ]
)

labels = range(1, 11)
plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(X[:, 0], X[:, 1], label="True Position")

for label, x, y in zip(labels, X[:, 0], X[:, 1]):
    plt.annotate(
        str(label),
        xy=(x, y),
        xytext=(-3, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
    )

plt.show()

linked = linkage(X, "single")
labelList = range(1, 11)
plt.figure(figsize=(10, 7))
dendrogram(
    linked,
    orientation="top",
    labels=labelList,
    distance_sort="decending",
    show_leaf_counts=True,
)
plt.show()

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
cluster.fit_predict(X)

plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')
plt.show() 

## ===================================================================================================

