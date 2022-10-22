#! usr/bin/env python3

from cgi import test
from cmath import pi
import numpy as np
import math
import scipy.stats as stats
import matplotlib.pyplot as plt

## Generaating data set from csv file

train_set = np.genfromtxt("hw04_data_set_train.csv", delimiter = ",")
test_set  = np.genfromtxt("hw04_data_set_test.csv", delimiter = ",")


X_train = train_set[:,0]
Y_train = train_set[:,1]
X_test = test_set[:,0]
Y_test = test_set[:,1]

N_train = train_set.shape[0]
N_test = test_set.shape[0]

## Train Regrossogram

bin_width = 0.1
minimum_value = 0
maximum_value = 2
data_interval = np.linspace(minimum_value, maximum_value, 2001)

left_borders = np.arange(minimum_value, maximum_value, bin_width)
right_borders = left_borders + bin_width

p_hat_nom = np.asarray([np.sum(((left_borders[b] < X_train) & (X_train <= right_borders[b]))* Y_train) for b in range(len(left_borders))])
p_hat_denom = np.asarray([np.sum((left_borders[b] < X_train) & (X_train <= right_borders[b])) for b in range(len(left_borders))])

regressogram = p_hat_nom/ p_hat_denom


plt.figure(figsize = (10,6))
plt.ylim([-1,2])
plt.xlabel("Time(sec)")
plt.ylabel("Signal (milivolt)")
plt.plot(X_train, Y_train, "b.")
plt.legend(['training'])
for b in range (len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [regressogram[b], regressogram[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [regressogram[b], regressogram[b+1]], "k-")

plt.figure(figsize = (10,6))
plt.ylim([-1,2])
plt.xlabel("Time(sec)")
plt.ylabel("Signal (milivolt)")
plt.plot(X_test, Y_test, "r.")
plt.legend(['test'])
for b in range (len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [regressogram[b], regressogram[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [regressogram[b], regressogram[b+1]], "k-")
plt.show()


## RMSE for Regressogram

rmse = 0 

for i in range(len(X_test)):
    for b in range(len(left_borders)):
        if ((left_borders[b] < X_test[i]) & (X_test[i] <= right_borders[b])):
            rmse += (Y_test[i]-regressogram[b])**2

rmse /= N_test
rmse = math.sqrt(rmse)

print("Regressogram => Rmse is " + str(rmse) + " when h is " + str(bin_width))

## Mean Smoother

bin_width = 0.1

p_hat_nom = np.asarray([np.sum((((x - 0.5 * bin_width) < X_train) & (X_train <=(x + 0.5 * bin_width))) * Y_train ) for x in data_interval])
p_hat_denom = np.asarray([np.sum(((x - 0.5 * bin_width) < X_train) & (X_train <= (x + 0.5*bin_width))) for x in data_interval])

running_mean_smoother = p_hat_nom/ p_hat_denom


plt.figure(figsize = (10,6))
plt.ylim([-1,2])
plt.xlabel("Time(sec)")
plt.ylabel("Signal (milivolt)")
plt.plot(X_train, Y_train, "b.")
plt.legend(['training'])
plt.plot(data_interval, running_mean_smoother, "k-")

plt.figure(figsize = (10,6))
plt.ylim([-1,2])
plt.xlabel("Time(sec)")
plt.ylabel("Signal (milivolt)")
plt.plot(X_test, Y_test, "r.")
plt.legend(['test'])
plt.plot(data_interval, running_mean_smoother, "k-")

plt.show()

## RMSE for Mean Smoother

rmse = 0 

for i in range(len(X_test)):
    for b in range(len(data_interval) -1 ):
        if ((data_interval[b] < X_test[i]) & (X_test[i] <= data_interval[b + 1])):
            rmse += (Y_test[i]- running_mean_smoother[b])**2

rmse /= N_test
rmse = math.sqrt(rmse)

print("Running mean smoother => Rmse is " + str(rmse) + " when h is " + str(bin_width))

## Kernel smoother

bin_width = 0.02


p_hat_nom = np.asarray([ np.sum(    (np.sqrt(1/(2*math.pi)) *np.exp(    (-0.5*(((x - X_train)/bin_width )**2))  ))*Y_train  )
                            for x in data_interval])

p_hat_denom = np.asarray([ np.sum(    np.sqrt(1/(2*math.pi)) *np.exp(    (-0.5*(((x - X_train)/bin_width )**2))  ) )
                            for x in data_interval])

kernel_smoother = p_hat_nom/p_hat_denom


plt.figure(figsize = (10,6))
plt.ylim([-1,2])
plt.xlabel("Time(sec)")
plt.ylabel("Signal (milivolt)")
plt.plot(X_train, Y_train, "b.")
plt.legend(['training'])
plt.plot(data_interval, kernel_smoother, "k-")

plt.figure(figsize = (10,6))
plt.ylim([-1,2])
plt.xlabel("Time(sec)")
plt.ylabel("Signal (milivolt)")
plt.plot(X_test, Y_test, "r.")
plt.legend(['test'])
plt.plot(data_interval, kernel_smoother, "k-")
plt.show()


## RMSE for Kernel Smoother

rmse = 0 

for i in range(len(X_test)):
    for b in range(len(data_interval) - 1):
        if ((data_interval[b] < X_test[i]) & (X_test[i] <= data_interval[b + 1])):
            rmse += (Y_test[i]- kernel_smoother[b])**2

rmse /= N_test
rmse = math.sqrt(rmse)

print("Kernel Smoother => Rmse is " + str(rmse) + " when h is " + str(bin_width))