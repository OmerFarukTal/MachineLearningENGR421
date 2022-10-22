#! usr/bin/env/ python3

import numpy as np
import pandas as pd
import cvxopt as cvx
import matplotlib.pyplot as plt

## Generating the data set

Images_data_set = np.genfromtxt("hw06_data_set_images.csv", delimiter = ",")
Labels_data_set = np.genfromtxt("hw06_data_set_labels.csv").astype(int)

X_train = Images_data_set[0:1000,:]
Y_train = Labels_data_set[0:1000]

X_test  = Images_data_set[1000:2000,:]
Y_test  = Labels_data_set[1000:2000]

N_train = X_train.shape[0]
D = X_train.shape[1]
N_test = X_test.shape[0]

print(X_train.shape)
print(Y_train.shape)

## Histogram creation


bin_width = 4
minimum_value = 0
maximum_value = 256

left_borders = np.arange(minimum_value, maximum_value, bin_width)
right_borders = left_borders + bin_width


p_hat_train = np.asarray([np.asarray([np.sum((left_borders[b] <= X_train[l,:]) & (X_train[l,:] < right_borders[b]))for b in range(len(left_borders))]) for l in range(N_train)])
histogram_train = p_hat_train/ (D ) # maybe * with bin_width


p_hat_test = np.asarray([np.asarray([np.sum((left_borders[b] <= X_test[l,:]) & (X_test[l,:] < right_borders[b]))for b in range(len(left_borders))]) for l in range(N_train)])
histogram_test = p_hat_test/ (D  ) # maybe * with bin_Width

print("Burası Histogram")
print(histogram_train[0:5,0:5])
print(histogram_test[0:5,0:5])


## Histogram Intersection Kernell


def intersection_kernel(H1, H2):
    K = []
    for i in range(H1.shape[0]):
        K_i = []
        for j in range(H2.shape[0]):
            K_i.append(np.sum(np.minimum(H1[i], H2[j])))
        K.append(K_i)
    return K
    

K_train = np.array(intersection_kernel(histogram_train, histogram_train))
print(K_train[0:5, 0:5])

K_test = np.array(intersection_kernel(histogram_test, histogram_train))
print(K_test[0:5, 0:5])

## Training kernel machine

def train_kernel_machine(C):
    yyK = np.matmul(Y_train[:,None], Y_train[None, :]) *K_train
    epsilon = 0.001

    P = cvx.matrix(yyK)
    q = cvx.matrix(-np.ones((N_train, 1)))
    A = cvx.matrix(1.0 * Y_train[None, :])
    b = cvx.matrix(0.0)
    G = cvx.matrix(np.vstack((-np.eye(N_train), np.eye(N_train))))
    h = cvx.matrix(np.vstack((np.zeros((N_train, 1)), C*np.ones((N_train, 1)))))


    result = cvx.solvers.qp(P, q, G, h, A, b)
    alpha = np.reshape(result["x"], N_train)

    alpha[alpha < C * epsilon ] = 0
    alpha[alpha > C * (1 - epsilon)] = C


    support_indices, = np.where(alpha != 0)
    active_indices, = np.where(np.logical_and(alpha!= 0, alpha < C))

    w0 = np.mean(Y_train[active_indices] * (1 -np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha[support_indices])))

    f_predicted = np.matmul(K_train, Y_train[:, None] * alpha[:,None]) + w0
    y_predicted = 2 *(f_predicted > 0.0) - 1

    y_predicted = np.resize(y_predicted, N_train)

    f_predicted_test = np.matmul(K_test, Y_train[:,None] * alpha[:,None]) + w0
    y_predicted_test = 2*(f_predicted_test > 0.0 ) - 1

    y_predicted_test = np.resize(y_predicted_test, N_test)

    return y_predicted, y_predicted_test
    

y_predicted, y_predicted_test = train_kernel_machine(10)

confusion_matrix_train = pd.crosstab(y_predicted, Y_train, rownames=['y_pred'], colnames=['y_truth'] )
print(confusion_matrix_train)

confusion_matrix_test = pd.crosstab(y_predicted_test, Y_test, rownames=['y_pred'], colnames=['y_truth'] )
print(confusion_matrix_test)


## Accuracy Graph

accuracy_list_train = []
accuracy_list_test  = []
data_set = []

for i in range(9):
    y_predicted_i, y_predicted_test_i = train_kernel_machine(10**(0.5*i + -1))
    data_set.append(0.5*i + -1)
    accuracy_list_train.append(np.sum(Y_train == y_predicted_i)/N_train)
    accuracy_list_test.append(np.sum(Y_test == y_predicted_test_i)/N_test)

print(accuracy_list_train)


plt.figure(figsize=(10,6))
plt.xlabel("Regularization parameter (C)")
plt.ylabel("Accuracy")
plt.plot(data_set, accuracy_list_train, "b-o")
plt.plot(data_set, accuracy_list_test, "r-o")
plt.legend(['training', 'test'])
plt.show()

