#! usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt 
import scipy.stats as stats

np.random.seed(732)

### Class mean, variance and sample numbers

class_mean1 = np.array([+0.0, 4.5])

class_mean2 = np.array([-4.5, -1.0])

class_mean3 = np.array([+4.5, -1.0])

class_mean4 = np.array([+0.0, -4.0])

class_variance1 = np.array([[+3.2, +0.0],
                            [+0.0, +1.2]])
                            
class_variance2 = np.array([[+1.2, +0.8],
                            [+0.8, +1.2]])

class_variance3 = np.array([[+1.2, -0.8],
                            [-0.8, +1.2]])

class_variance4 = np.array([[+1.2, +0.0],
                            [+0.0, +3.2]])

N1 = 105
N2 = 145
N3 = 135
N4 = 115

### Generate points

points1 = np.random.multivariate_normal(class_mean1, class_variance1, N1)
points2 = np.random.multivariate_normal(class_mean2, class_variance2, N2)
points3 = np.random.multivariate_normal(class_mean3, class_variance3, N3)
points4 = np.random.multivariate_normal(class_mean4, class_variance4, N4)

points = np.concatenate((points1, points2, points3, points4))

### Plot the generated points

"""
plt.figure(figsize = (10,10))
plt.xlabel('$x1$')
plt.ylabel('$x2$')
plt.axis([-8,8,-8,8])
plt.plot(points1[:,0], points1[:,1], "r.", markersize = 10)
plt.plot(points2[:,0], points2[:,1], "g.", markersize = 10)
plt.plot(points3[:,0], points3[:,1], "b.", markersize = 10)
plt.plot(points4[:,0], points4[:,1], "m.", markersize = 10)
plt.show()
"""

### I just took transpose, and total sample

sample_points1 = points1.T
sample_points2 = points2.T
sample_points3 = points3.T
sample_points4 = points4.T

N = N1 + N2 + N3 + N4

### Getting sample mean

sample_mean1 = np.mean(sample_points1, axis = 1)
sample_mean2 = np.mean(sample_points2, axis = 1)
sample_mean3 = np.mean(sample_points3, axis = 1)
sample_mean4 = np.mean(sample_points4, axis = 1)

sample_means = np.array([sample_mean1, sample_mean2, sample_mean3, sample_mean4])
print(sample_means)


class_priors = np.array([N1/N, N2/N, N3/N, N4/N])
print(class_priors)

sample_covariance1 = np.cov(sample_points1)
sample_covariance2 = np.cov(sample_points2)
sample_covariance3 = np.cov(sample_points3)
sample_covariance4 = np.cov(sample_points4)

sample_covariances = np.array([sample_covariance1, sample_covariance2, sample_covariance3, sample_covariance4])
print(sample_covariances)


### Creating the scoring functions

Nelem = 161

data_interval_X = np.linspace(-8, 8, Nelem)
data_interval_Y = np.linspace(-8, 8, Nelem)

data_interval = np.array([[[data_interval_X[k], data_interval_Y[l]] for k in range(Nelem) ] for l in range(Nelem)]).reshape(-1,2)


#print(data_interval)
"""
plt.figure(figsize = (10,10))
plt.xlabel('$x1$')
plt.ylabel('$x2$')
plt.axis([-8,8,-8,8])
plt.plot(data_interval[:,0], data_interval[:,1], "m.", markersize = 10)
plt.show()
"""
# Scoring the points


scoring_points = np.array([[
    -0.5*np.dot(np.dot(points[k,:].reshape(1,2), np.linalg.inv(sample_covariances[l])), points[k,:].reshape(2,1))
    + np.dot(  np.dot(np.linalg.inv(sample_covariances[l]), sample_means[l].reshape(2,1)).T, points[k,:].reshape(2,1) )
    - 0.5*np.dot(np.dot(sample_means[l].reshape(1,2),np.linalg.inv(sample_covariances[l])) , sample_means[l].reshape(2,1))
    - 1*np.log(2*math.pi)
    - 0.5*np.log(np.linalg.det(sample_covariances[l]))
    + np.log(class_priors[[l]])
        
    for k in range(N)] 
    for l in range (4)]).reshape(4,-1)

class_assignment_points = np.argmax(scoring_points, axis = 0)

#print(class_assignment_points)

### Confusio matrix

classes = np.concatenate(  (np.full((N1,1),0), np.full((N2,1),1), np.full((N3,1),2), np.full((N4,1),3)) ).reshape(1,N)[0]

confusion_matrix = np.array([[points[(classes == 0) & (class_assignment_points == 0)].shape[0], points[(classes == 1) & (class_assignment_points == 0)].shape[0], points[(classes == 2) & (class_assignment_points == 0)].shape[0], points[(classes == 3) & (class_assignment_points == 0)].shape[0]],
                             [points[(classes == 0) & (class_assignment_points == 1)].shape[0], points[(classes == 1) & (class_assignment_points == 1)].shape[0], points[(classes == 2) & (class_assignment_points == 1)].shape[0], points[(classes == 3) & (class_assignment_points == 1)].shape[0]],
                             [points[(classes == 0) & (class_assignment_points == 2)].shape[0], points[(classes == 1) & (class_assignment_points == 2)].shape[0], points[(classes == 2) & (class_assignment_points == 2)].shape[0], points[(classes == 3) & (class_assignment_points == 2)].shape[0]],
                             [points[(classes == 0) & (class_assignment_points == 3)].shape[0], points[(classes == 1) & (class_assignment_points == 3)].shape[0], points[(classes == 2) & (class_assignment_points == 3)].shape[0], points[(classes == 3) & (class_assignment_points == 3)].shape[0]]])

print(confusion_matrix)

### Scoring functions

scoring = np.array([[
    -0.5*np.dot(np.dot(data_interval[k,:].reshape(1,2), np.linalg.inv(sample_covariances[l])), data_interval[k,:].reshape(2,1))
    + np.dot(  np.dot(np.linalg.inv(sample_covariances[l]), sample_means[l].reshape(2,1)).T, data_interval[k,:].reshape(2,1) )
    - 0.5*np.dot(np.dot(sample_means[l].reshape(1,2),np.linalg.inv(sample_covariances[l])) , sample_means[l].reshape(2,1))
    - 1*np.log(2*math.pi)
    - 0.5*np.log(np.linalg.det(sample_covariances[l]))
    + np.log(class_priors[[l]])
        
    for k in range(Nelem*Nelem)] 
    for l in range (4)]).reshape(4,-1)


#print(scoring)
class_assignment = np.argmax(scoring, axis=0)


### Marking and plotting points

mislabeled_points = points[((classes == 0) & (class_assignment_points != 0) |
                            (classes == 1) & (class_assignment_points != 1) |
                            (classes == 2) & (class_assignment_points != 2) |
                            (classes == 3) & (class_assignment_points != 3)
                            )]

plt.figure(figsize = (10,10))
plt.xlabel('$x1$')
plt.ylabel('$x2$')
plt.axis([-8,8,-8,8])
plt.plot(data_interval[class_assignment == 0][:,0], data_interval[class_assignment == 0][:,1] ,"r.", markersize = 1)
plt.plot(data_interval[class_assignment == 1][:,0], data_interval[class_assignment == 1][:,1] ,"g.", markersize = 1)
plt.plot(data_interval[class_assignment == 2][:,0], data_interval[class_assignment == 2][:,1] ,"b.", markersize = 1)
plt.plot(data_interval[class_assignment == 3][:,0], data_interval[class_assignment == 3][:,1] ,"m.", markersize = 1)
plt.plot(points1[:,0], points1[:,1], "r.", markersize = 10)
plt.plot(points2[:,0], points2[:,1], "g.", markersize = 10)
plt.plot(points3[:,0], points3[:,1], "b.", markersize = 10)
plt.plot(points4[:,0], points4[:,1], "m.", markersize = 10)
plt.plot(mislabeled_points[:,0], mislabeled_points[:,1], "k.", 'o', mfc = 'none', markersize = 20)
plt.show()
