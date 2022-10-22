#! usr/bin/env/ python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import scipy.spatial as spa
import math

## Generating data from csv file 

X = np.genfromtxt("hw08_data_set.csv", delimiter = ",")
centroids = np.genfromtxt("hw08_initial_centroids.csv", delimiter = ",")

N = X.shape[0]
D = X.shape[1]
K = 9

original_centroids = np.array([[+5.0, +5.0],
                                [-5.0, +5.0],
                                [-5.0, -5.0],
                                [+5.0, -5.0],
                                [+5.0,  0.0],
                                [ 0.0, +5.0],
                                [-5.0,  0.0],
                                [ 0.0, -5.0],
                                [ 0.0,  0.0]])

original_covariances = np.array([  [[+0.8, -0.6],
                                    [-0.6, +0.8]], #1
                                   [[+0.8, +0.6],
                                    [+0.6, +0.8]], #2
                                   [[+0.8, -0.6],
                                    [-0.6, +0.8]], #3
                                   [[+0.8, +0.6],
                                    [+0.6, +0.8]], #4
                                   [[+0.2, +0.0],
                                    [+0.0, +1.2]], #5
                                   [[+1.2, +0.0], 
                                    [+0.0, +0.2]], #6
                                   [[+0.2, +0.0], 
                                    [+0.0, +1.2]], #7
                                   [[+1.2, +0.0], 
                                    [+0.0, +0.2]], #8
                                   [[+1.6, +0.0], 
                                    [+0.0, +1.6]], #9
                                    ])


## Update centroids, covariances, memberships

def update_centroids(memberships, X):
    if memberships is None:
        return centroids
    else:
        #centroids = np.vstack([np.mean(X[memberships == k,:], axis = 0) for k in range(9)])
        for k in range(K):
            coeffiecients = np.reshape( memberships[:,k], (N,1))
            coeffiecient_matrix = np.hstack([coeffiecients for i in range(D)])
            centroids[k] = np.sum(coeffiecient_matrix*X, axis = 0) / np.sum(coeffiecients)
        return centroids

def update_covariances(memberships, X):
    # return np.vstack([np.cov(X[memberships == k,:]) for k in range(K) ])
    covariances = np.zeros((K,D,D))
    for k in range(K):
        coeffiecients = np.reshape( memberships[:,k], (N,1))
        coeffiecient_matrix = np.hstack([coeffiecients for i in range(D)])
        mult_matrix = coeffiecient_matrix*X

        coefficient_member = np.matmul(np.reshape(memberships[:,k], (N,1)), np.reshape(centroids[k], (1,D)))

        mult_matrix -= coefficient_member
        mult_matrix = mult_matrix.T

        second = X - np.vstack([centroids[k] for i in range(N)])
        covariances[k] = np.matmul(mult_matrix, second)/ np.sum(coeffiecients)

    return covariances

def update_cluster_prob(memberships, X):
    return np.sum(memberships, axis = 0)/N

def update_memberships(centroids, covariances, cluster_prob, memberships,X):
    if memberships is None:
        D = spa.distance_matrix(centroids, X).T
        mins = np.argmin(D, axis = 1)
        memberships = []
        for i in range(N):
            memberships.append(np.zeros((1,K)))
            memberships[i][0,:][mins[i]] = 1
        memberships = np.array(memberships)[:,0,:] #### YANLIŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞŞ
        return memberships
    else:
        # stats.multivariate_normal.pdf()
        memberships = np.zeros((N,K))
        for k in range(K):
            memberships[:,k] = stats.multivariate_normal.pdf(X, centroids[k], covariances[k])*cluster_prob[k]
        for i in range(1000):
            memberships[i,:] = memberships[i,:] / np.sum(memberships[i,:])
        return memberships


memberships1 = None
covariances = None
cluster_prob = None


## First 100 Iteration Report

iteration = 1

for i in range(100):
    old_centroids = centroids
    centroids = update_centroids(memberships1, X)

    old_memberships = memberships1
    memberships1 =  update_memberships(centroids, covariances, cluster_prob, memberships1,X)

    covariances = update_covariances(memberships1, X)
    cluster_prob = update_cluster_prob(memberships1, X)

    
print(centroids)

Y_train = np.argmax(memberships1, axis = 1)

# Drawin the graph

x1 = np.linspace(-8,+8,1001)
x2 = np.linspace(-8,+8,1001)

x1_grid, x2_grid = np.meshgrid(x1,x2)

positions = np.vstack([x1_grid.ravel(), x2_grid.ravel()])
grid_positions = (np.array(positions)).T

point_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6"])

def gaussian(grid_position, covariance, centroid):

    G = stats.multivariate_normal.pdf(grid_position,covariance, centroid)

    return G


plt.figure(figsize = (10,10))
plt.xlim(-8,+8)
plt.ylim(-8,+8)
for c in range(K):
    plt.plot(X[Y_train == c ,0], X[Y_train == c ,1], marker = "o",
               markersize = 4, linestyle = "none", color = point_colors[c])
    G1  = gaussian(grid_positions, centroids[c], covariances[c])
    plt.plot(grid_positions[(G1 > 0.049) & (G1 < 0.051) ][:,0], grid_positions[(G1 > 0.049) & (G1 < 0.051) ][:,1], marker = "s", markersize = 2,linestyle = "none",color = point_colors[c] )
    G2 = gaussian(grid_positions, original_centroids[c], original_covariances[c])
    plt.plot(grid_positions[(G2 > 0.049) & (G2 < 0.051) ][:,0], grid_positions[(G2 > 0.049) & (G2 < 0.051) ][:,1], marker = "p", markersize = 2,linestyle = "none",color = "black" )
plt.show()
