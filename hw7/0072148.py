#! usr/bin/env/ python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import scipy.spatial.distance as dt

## Generating data from the csv file

Images = np.genfromtxt("hw07_data_set_images.csv", delimiter = ",")
Labels = np.genfromtxt("hw07_data_set_labels.csv").astype(int)

X_train = Images[:2000,:]
Y_train = Labels[:2000]

X_test = Images[2000:,:]
Y_test = Labels[2000:]

N_train = X_train.shape[0]
D = X_train.shape[1]
N_test = X_test.shape[0]
K = np.max(Y_train)


## Sw and Sb

sample_means = []

for c in range(1, K + 1):
    sample_means.append(np.mean(X_train[Y_train == c], axis= 0))

sample_means = np.array(sample_means)

mean = np.mean(X_train, axis = 0)


Sw_c = []

for c in range(1, K+1):
    z1 = X_train - sample_means[c -1]
    z2 = z1.T
    Sw_c.append(np.cov(X_train[Y_train == c].T, ddof = 0)*len(X_train[Y_train == c]))

Sw_c = np.array(Sw_c)

Sw = np.sum(Sw_c,axis = 0)
print("Sw")
print(Sw[0:5,0:5])


Sb_c = []

for c in range(1, K +1):
    z1 = np.resize(sample_means[c -1] - mean, (D,1))
    z2 = np.resize(sample_means[c -1] - mean, (1,D))
    Sb_c.append(np.matmul( z1, z2 )*len(X_train[Y_train == c]))

Sb_c = np.array(Sb_c)

Sb = np.sum(Sb_c, axis = 0)
print("Sb : ")
print(Sb[0:5, 0:5])


## Eigen Values and Eigen vectors

Sw_Sb = np.matmul(np.linalg.inv(Sw), Sb)

values, vectors = np.linalg.eig(Sw_Sb)
values = np.real(values)
vectors = np.real(vectors)

print(values[0:9])

## Projection onto 2-dimensional subspace

Z_train = np.matmul(X_train - mean, vectors[:, [0, 1]])

point_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6"])

plt.figure(figsize = (10, 10))
plt.xlim(-6,6)
plt.ylim(-6,6)
for c in range(K):
    plt.plot(Z_train[Y_train == c + 1,0], Z_train[Y_train == c + 1,1], marker = "o",
                markersize = 4, linestyle = "none", color = point_colors[c])
plt.legend(["t-shirt/top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle bot"], loc= 'upper left')
plt.xlabel("Component#1")
plt.ylabel("Component#2")


Z_test = np.matmul(X_test - np.mean(X_test, axis = 0), vectors[:, [0, 1]])

plt.figure(figsize = (10, 10))
plt.xlim(-6,6)
plt.ylim(-6,6)
for c in range(K):
    plt.plot(Z_test[Y_test == c + 1,0], Z_test[Y_test == c + 1,1], marker = "o",
                markersize = 4, linestyle = "none", color = point_colors[c])
plt.legend(["t-shirt/top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle bot"], loc= 'upper left')
plt.xlabel("Component#1")
plt.ylabel("Component#2")
plt.show()


## Classifying with K-nearest Neighboor Classifier

k = 11

Z_train = np.matmul(X_train - mean, vectors[:, 0: 9])
Z_test  = np.matmul(X_test - np.mean(X_test, axis = 0), vectors[:, 0: 9])


def make_prediction(Z):
    distances = dt.cdist(Z, Z_train)
    sorted_distances = np.argsort(distances)
    labels = Y_train[sorted_distances[:,0:k]]
    class_assignment = np.array([Counter(row).most_common(1)[0][0] for row in labels])
    return class_assignment


y_predicted_train = make_prediction(Z_train)

confusion_matrix_train = pd.crosstab(y_predicted_train, Y_train, rownames=['y_hat'], colnames=['y_train'] )
print(confusion_matrix_train)


y_predicted_test = make_prediction(Z_test)

confusion_matrix_test = pd.crosstab(y_predicted_test, Y_test, rownames=['y_hat'], colnames=['y_test'] )
print(confusion_matrix_test)
