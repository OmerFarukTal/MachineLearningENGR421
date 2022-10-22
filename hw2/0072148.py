#! usr/bin/env python3

import numpy as np
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

def safelog(x):
    return np.log(x + 1e-100)

## Getting all the images and labels from csv file
all_images = np.genfromtxt("hw02_data_set_images.csv", delimiter= ",")
all_labels = np.genfromtxt("hw02_data_set_labels.csv").astype(int)

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

training_labels = np.concatenate((all_labels[0:25], all_labels[39:64], all_labels[78:103], all_labels[117:142], all_labels[156:181]))
training_set = np.concatenate((training_set_images_1, training_set_images_2, training_set_images_3, training_set_images_4, training_set_images_5))
test_labels = np.concatenate((all_labels[25:39], all_labels[64:78], all_labels[103:117], all_labels[142:156], all_labels[181:195]))
test_set = np.concatenate((test_set_images_1, test_set_images_2, test_set_images_3, test_set_images_4, test_set_images_5))


## Getting class number

K = np.max(training_labels)
N = 125
N_test = 70
## Calculate prior probabilities and class means together

class_priors = np.array([ np.mean(training_labels == c+1) for c in range(K)])

pcd = np.array([ np.mean(training_set[training_labels == c+1], axis = 0)  for c in range(K)])

print(pcd)
print(class_priors)


## Scoring function for trainig set

"""
data_point_1 = training_set[0,:]
#print(data_point_1)
print(pcd[0,:].shape)

scoring_data_point_1 =  np.dot(safelog(pcd[0,:]).T,data_point_1 ) + np.dot(safelog(1 - pcd[0,:]).T, 1- data_point_1)  +  np.log(class_priors[0])
scoring_data_point_2 =  np.dot(safelog(pcd[1,:]).T,data_point_1 ) + np.dot(safelog(1 - pcd[1,:]).T, 1- data_point_1)  +  np.log(class_priors[1])
scoring_data_point_3 =  np.dot(safelog(pcd[2,:]).T,data_point_1 ) + np.dot(safelog(1 - pcd[2,:]).T, 1- data_point_1)  +  np.log(class_priors[2])
scoring_data_point_4 =  np.dot(safelog(pcd[3,:]).T,data_point_1 ) + np.dot(safelog(1 - pcd[3,:]).T, 1- data_point_1)  +  np.log(class_priors[3])
scoring_data_point_5 =  np.dot(safelog(pcd[4,:]).T,data_point_1 ) + np.dot(safelog(1 - pcd[4,:]).T, 1- data_point_1)  +  np.log(class_priors[4])


print(scoring_data_point_1)
print(scoring_data_point_2)
print(scoring_data_point_3)
print(scoring_data_point_4)
print(scoring_data_point_5)
"""

scoring_training_points = np.stack([ [ 
                    np.dot(safelog(pcd[c,:]).T,training_set[k,:]) + np.dot( safelog(1 - pcd[c,:]).T, 1 - training_set[k,:]) + np.log(class_priors[c])
    for k in range(N) ]
    for c in range(K)
])

predicted_labels = np.argmax(scoring_training_points, axis = 0)

df_confusion = pd.crosstab(predicted_labels + 1, training_labels, rownames= ['y_pred'], colnames=['y_truth'])

print(df_confusion)

## Scoring function for test set

scoring_test_points = np.stack([[
            np.dot(test_set[k,:], safelog(pcd[c,:]).T) + np.dot(1 - test_set[k,:], safelog(1 -pcd[c,:]).T) + np.log(class_priors[c])
    for k in range(N_test)]
    for c in range(K)
])

predicted_labels_test_set = np.argmax(scoring_test_points, axis = 0)

df_confusion_test = pd.crosstab(predicted_labels_test_set + 1, test_labels, rownames=['y_pred'], colnames=['y_truth'] )

print(df_confusion_test)
