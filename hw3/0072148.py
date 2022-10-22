#! usr/bin/env python3

from matplotlib import markers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


## Getting all the images and labels from csv file
all_images = np.genfromtxt("hw03_data_set_images.csv", delimiter= ",")
all_labels = np.genfromtxt("hw03_data_set_labels.csv").astype(int)

## Grouping all traning set and test sets together
training_set_images_1 = all_images[0:25,:]
test_set_images_1 = all_images[25:39,:]
training_set_images_2 = all_images[39:64,:]
test_set_images_2 = all_images[64:78,:]
training_set_images_3 = all_images[78:103,:]
test_set_images_3 = all_images[103:117,:]
training_set_images_4 = all_images[117:142,:]
test_set_images_4 = all_images[142:156,:]
training_set_images_5 = all_images[156:181,:]
test_set_images_5 = all_images[181:195,:]

## Initializing all traininig labels, and training set 

training_labels = np.concatenate((all_labels[0:25], all_labels[39:64], all_labels[78:103], all_labels[117:142], all_labels[156:181])).astype(int)
training_set = np.concatenate((training_set_images_1, training_set_images_2, training_set_images_3, training_set_images_4, training_set_images_5))
test_labels = np.concatenate((all_labels[25:39], all_labels[64:78], all_labels[103:117], all_labels[142:156], all_labels[181:195])).astype(int)
test_set = np.concatenate((test_set_images_1, test_set_images_2, test_set_images_3, test_set_images_4, test_set_images_5))

K = np.max(training_labels)
D = training_set.shape[1]
N = training_set.shape[0]

Y_truth = np.zeros((N,K)).astype(int)
Y_truth[range(N), training_labels -1] = 1


## Initalization of parameters

W = np.random.uniform(low = -0.01, high = +0.01, size = (D, K))
w0 = np.random.uniform(low = -0.01, high = +0.01, size = (1,K))


## Sigmoid function

def sigmoid(X, W, w0):

    return 1 / (1 + np.exp(-np.matmul(np.hstack(( X, np.ones((X.shape[0],1)))), np.vstack((W,w0)))))

## Gradients

def gradient_W(X, y_truth, y_pred):
    
    return -np.asarray([np.matmul(((y_truth[:,c] - y_pred[:,c])*(1-y_pred[:,c])*(y_pred[:,c])), X )for c in range(K)]).T


def gradient_w0(y_truth, y_pred):
    
    return -np.sum(((y_truth-y_pred)*(1-y_pred)*(y_pred)),axis = 0)

"""
y_pred_deneme = sigmoid(training_set,W,w0)
res = gradient_w0(Y_truth, y_pred_deneme)

print(res)
"""
"""
y_pred_deneme = sigmoid(training_set,W,w0)
res2 = gradient_W(training_set, Y_truth, y_pred_deneme)
print(res2.shape)
"""

## Initialization eta and epsilon

eta = 0.001
epsilon = 0.001

## Optimazing

iteration = 1
objective_values = []

while True:
    y_prediction = sigmoid(training_set, W, w0)

    objective_values = np.append(objective_values, 
                            -np.sum((-1/2)*(Y_truth -y_prediction)**2))

    w0_old = w0
    W_old = W

    w0 = w0 - eta*gradient_w0(Y_truth,y_prediction)
    W = W - eta*gradient_W(training_set, Y_truth, y_prediction)

    if (np.sqrt(np.sum((w0-w0_old)**2 + np.sum(W -W_old)**2)) < epsilon):
        break

    iteration = iteration + 1

## W and w0

print(W)
print(w0)


## Convergence

plt.figure(figsize=(8,4))
plt.plot(range(1,iteration + 1), objective_values, "k")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()


## Confusion matrices

training_set_classification = np.argmax(y_prediction, axis = 1).T

df_confusion_test = pd.crosstab(training_set_classification + 1, training_labels, rownames=['y_pred'], colnames=['y_truth'] )
print(df_confusion_test)

test_set_pred = sigmoid(test_set, W, w0)
test_set_classification = np.argmax(test_set_pred, axis = 1)

df_confusion_test = pd.crosstab(test_set_classification + 1, test_labels, rownames=['y_pred'], colnames=['y_truth'] )
print(df_confusion_test)
