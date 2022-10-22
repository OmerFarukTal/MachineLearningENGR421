#! usr/bin/env python3

from matplotlib import markers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

## Generating data points from csv files

Train_data_set = np.genfromtxt("hw05_data_set_train.csv", delimiter = ",")
Test_data_set = np.genfromtxt("hw05_data_set_test.csv", delimiter = ",")

X_train = Train_data_set[:,0]
Y_train = Train_data_set[:,1]

X_test = Test_data_set[:,0]
Y_test = Test_data_set[:,1]

## Initalize P, N, D 

N_train = X_train.shape[0]
N_test = X_test.shape[0]

P = 30

def train_tree(X_train, Y_train, P_):

    ## Decision tree algorithm initializion

    node_indices = {} # keep track of datat points enter node
    is_terminal = {} # if node is terminal or not
    need_split = {} # if a further split is needed or not

    node_features = {} # actually we have one feature but I want to keep it
    node_splits = {} #this is for w0 

    gm_values = {}

    node_indices[1] = np.array(range(N_train))
    is_terminal[1] = False
    need_split[1] = True



    ## Tree Inference



    while True:
        split_nodes = [key for key,value in need_split.items() if value == True]
        
        if len(split_nodes) == 0: 
            break

        for split_node in split_nodes:
            data_indices = node_indices[split_node]
            need_split[split_node] = False
            
            if len(np.unique(Y_train[data_indices])) < P_:   ##### Check here !!!!!
                is_terminal[split_node] = True
                gm_values[split_node] = np.sum(Y_train[data_indices]) / len(data_indices)

            else:
                is_terminal[split_node] = False

                best_scores = 0.0
                best_splits = 0.0
                
                unique_values = np.sort(np.unique(X_train[data_indices]))

                split_positions = ( unique_values[1: len(unique_values)] + \
                                    unique_values[0:(len(unique_values)-1)] ) /2

                split_scores = np.repeat(0.0, len(split_positions))

                for s in range(len(split_positions)):

                    left_indices = data_indices[X_train[data_indices] > \
                                                split_positions[s]]

                    right_indices = data_indices[X_train[data_indices] <= \
                                                    split_positions[s]]

                    #g_m = np.sum(Y_train[data_indices]) / len(data_indices)

                    left_gm = np.sum(Y_train[left_indices]) / len(left_indices)
                    right_gm = np.sum(Y_train[right_indices]) / len(right_indices)

                    split_scores[s] = (1/len(data_indices)) * np.sum( (Y_train[left_indices] - left_gm)**2 ) +\
                                    (1/len(data_indices)) * np.sum( (Y_train[right_indices] - right_gm)**2)

                    best_scores = np.min(split_scores)
                    best_splits = split_positions[np.argmin(split_scores)] # best w_m0 for dth feature

                split_d = best_scores

                node_features[split_node] = split_d
                node_splits[split_node] = best_splits

                left_indices = data_indices[X_train[data_indices] > \
                                                    best_splits]

                node_indices[2 * split_node] = left_indices
                is_terminal[2 * split_node] = False
                need_split[2 * split_node] = True

                right_indices = data_indices[X_train[data_indices] <= \
                                                    best_splits]

                node_indices[2 * split_node + 1] = right_indices
                is_terminal[2 * split_node + 1] = False
                need_split[2 * split_node + 1] = True

    return node_indices, is_terminal, need_split, node_features, node_splits, gm_values
## Drawing graphs 

node_indices, is_terminal, need_split, node_features, node_splits, gm_values = train_tree(X_train, Y_train, P)

train_dic = {}

for key,value in node_indices.items():
    for v in value:
        if key in gm_values:
            if X_train[v] not in train_dic:
                train_dic[X_train[v]] = gm_values[key]

train_dic = {k: v for k, v in sorted(train_dic.items(), key=lambda item: item[0])}


plt.figure(figsize = (10,6))
plt.ylim([-1,2])
plt.xlabel("Time(sec)")
plt.ylabel("Signal (milivolt)")
plt.plot(X_train, Y_train, "b.")
plt.legend(['training'])
for b in range(1,len(train_dic)):
    plt.plot([list(train_dic.keys())[b-1], list(train_dic.keys())[b]], [list(train_dic.values())[b], list(train_dic.values())[b]] , "k-")
for b in range(len(train_dic) -1):
    plt.plot([list(train_dic.keys())[b], list(train_dic.keys())[b]], [list(train_dic.values())[b], list(train_dic.values())[b + 1]] , "k-")

plt.figure(figsize = (10,6))
plt.ylim([-1,2])
plt.xlabel("Time(sec)")
plt.ylabel("Signal (milivolt)")
plt.plot(X_test, Y_test, "r.")
plt.legend(['test'])
for b in range(1,len(train_dic)):
    plt.plot([list(train_dic.keys())[b-1], list(train_dic.keys())[b]], [list(train_dic.values())[b], list(train_dic.values())[b]] , "k-")
for b in range(len(train_dic) -1):
    plt.plot([list(train_dic.keys())[b], list(train_dic.keys())[b]], [list(train_dic.values())[b], list(train_dic.values())[b + 1]] , "k-")

plt.show()


## Rmse error for Train and test dataset

def RMSE_Error(is_terminal, gm_values, node_splits,node_indices, X_train ,Y_train, Y_test):

    train_dic = {}

    for key,value in node_indices.items():
        for v in value:
            if key in gm_values:
                if X_train[v] not in train_dic:
                    train_dic[X_train[v]] = gm_values[key]

    train_dic = {k: v for k, v in sorted(train_dic.items(), key=lambda item: item[0])}

    sum1 = 0.0
    for i in range(len(Y_train)):
        sum1 += (Y_train[i] - train_dic[X_train[i]])**2

    sum1 /= N_train
    sum1 = math.sqrt(sum1)


    sum2 = 0.0
    y_predicted_test = np.repeat(0.0, N_test)
    for i in range(N_test):
        index = 1
        while True:
            if is_terminal[index] == True:
                y_predicted_test[i] = gm_values[index]
                break
            else:
                if X_test[i] > node_splits[index]:
                    index = index * 2
                else:
                    index = index * 2 + 1


    sum2 = np.sum( (Y_test - y_predicted_test)**2 )
    sum2 /= N_test
    sum2 = math.sqrt(sum2)
    
    return sum1, sum2

rmse_train, rmse_test = RMSE_Error(is_terminal,gm_values,node_splits, node_indices,X_train, Y_train, Y_test) 


print(f"RMSE on training set is {rmse_train} when P is {P}")
print(f"RMSE on training set is {rmse_test} when P is {P}")

rmse_trains = []
rmse_tests = []
Ps = []

for i in range(10, 51, 5):
    node_indices_1, is_terminal_1, need_split_1 , node_features_1 , node_splits_1 , gm_values_1 = train_tree(X_train, Y_train, i)
    rmse_train_1, rmse_test_1 = RMSE_Error(is_terminal_1,gm_values_1,node_splits_1,node_indices_1, X_train, Y_train, Y_test) 
    rmse_trains.append(rmse_train_1)
    rmse_tests.append(rmse_test_1)
    Ps.append(i)


plt.figure(figsize = (10,6))
plt.xlabel("Pre-Pruning size (P)")
plt.ylabel("RMSE")
plt.plot(Ps, rmse_trains,"b-o")
plt.plot(Ps, rmse_tests, "r-o" )
plt.legend(['training','test'])
plt.show()
