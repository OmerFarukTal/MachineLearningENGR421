#! usr/bin/env/ python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import scipy.spatial as spa
import scipy.spatial.distance as dt
from scipy.linalg import sqrtm
import math


## Generating data from csv file 

X = np.genfromtxt("hw09_data_set.csv", delimiter = ",")

N = X.shape[0]
D = X.shape[1]
d = 2.0

## Initialize B matrix with d <= 2

distances = dt.cdist(X, X)
B = (distances < d).astype(int)
print((B == 1)[1,:].shape)
## Visualize the connectivity of data points

"""
plt.figure(figsize = (10,10))
plt.xlim((-8,+8))
plt.ylim((-8,+8))
plt.plot(X[:,0], X[:,1], markersize = 5, marker = "o", linestyle = "none" ,color = "black", zorder = 0)
for i in range(N):
    plt.plot(X[(B == 1)[i,:]][:,0], X[(B == 1)[i,:]][:,1], color = "#C0C0C0", alpha = 0.2, zorder = 10)
plt.show()
"""

D = np.diag(np.sum(B, axis = 1))
values, vectors  = np.linalg.eig(D)
values = np.real(values)
vectors = np.real(vectors)

D_sqrt = np.matmul(np.matmul(vectors.T , np.diag(np.sqrt(values))), np.linalg.inv(vectors.T))
D_minus_sqrt = np.linalg.inv(D_sqrt)
L_symmetric = np.eye(N) - np.matmul(np.matmul(D_minus_sqrt, B), D_minus_sqrt)
print(L_symmetric[0:5,0:5])



D_sqrt = sqrtm(D)
D_minus_sqrt = np.linalg.inv(D_sqrt)

L_symmetric = np.eye(N) - np.matmul(np.matmul(D_minus_sqrt, B), D_minus_sqrt)
print(L_symmetric[0:5,0:5])

## Eigenvalues of the Laplacian matrix
"""
values, vectors  = np.linalg.eig(L_symmetric)
values = np.real(values)
vectors = np.real(vectors)

R = 6

values_sorted_index = np.argsort(values)[1:R]
selected_vectors = vectors[values_sorted_index,:]

print(selected_vectors)
print(selected_vectors.shape)


Z = np.vstack([selected_vectors[i] for i in range (5)]).T
print(Z[0:5, 0:5])
print(Z.shape)
## K - means Clustering

K = 9


def update_centroids(membership, X):
    if membership is None:
        centroids = np.vstack([Z[242,:], Z[528,:], Z[570,:], Z[590,:], Z[648,:], Z[667,:], Z[774,:], Z[891,:], Z[955,:]])
        return centroids
    else:
        centroids = np.vstack([np.mean(X[membership == k,:], axis = 0)for k in range(K)])
        return centroids

def update_memberships(centroids, X):
    D = spa.distance_matrix(centroids, X)
    membership = np.argmin(D, axis = 0)
    return membership

memberships = None
centroids = None

for i in range(10000):
    old_centroids = centroids
    centroids = update_centroids(memberships, Z)

    #if np.alltrue(centroids == old_centroids):
    #    break

    old_memberships = memberships
    memberships = update_memberships(centroids, Z)

    #if np.alltrue(memberships == old_memberships):
    #    break

point_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6"])

plt.figure(figsize = (10,10))
plt.xlim(-8,+8)
plt.ylim(-8,+8)
for c in range(K):
    plt.plot(X[memberships == c ,0], X[memberships == c ,1], marker = "o",
               markersize = 4, linestyle = "none", color = point_colors[c])
    plt.plot(centroids[c, 0], centroids[c, 1], marker = "s", markersize = 7, markerfacecolor = point_colors[c], markeredgecolor = "black")
plt.show()












"""