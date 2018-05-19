import sklearn.preprocessing as pp
from sklearn import decomposition
from sklearn import cluster
import numpy as np


def transforms(data):
    scare = pp.MinMaxScaler(feature_range=(0, 1))
    return scare.fit_transform(data)

def data_pca(data, stand):
    pca = decomposition.PCA()
    pca.fit(data)
    data_weight = np.where(pca.explained_variance_ > stand, pca.explained_variance_, 0)
    return [i for i in range(len(data_weight)) if data_weight[i] == 0]

def kmeans_cluster(dataSet):
    k_means = cluster.KMeans(n_clusters=2)
    k_means.fit(dataSet)
    return k_means.labels_

