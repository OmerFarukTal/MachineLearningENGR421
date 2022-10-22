#! usr/bin/env python3
import numpy as np
import math
import scipy.stats as stats
import matplotlib.pyplot as plt

np.random.seed(421)

x_points = np.sort(np.random.random(20))
y_points = np.sort(np.random.random(20))*2 +1

print(x_points)
print(y_points)


plt.figure(figsize = (10,6))
plt.plot(x_points, y_points, "r.", markersize = 10)
plt.show()

"""
We created the data points
We can save them to somerwhere 
In the rest of the code I will assume that I took this data from somewhere else 
"""

# liner regression y = w0 + w1x

N = x_points.shape[0]

w1 = (np.dot(x_points.T, y_points) - (np.mean(x_points)*np.mean(y_points))*N)/ (np.dot(x_points.T,x_points) - N*np.mean(x_points)**2)
w0 = np.mean(y_points) - w1*(np.mean(x_points))

print(w0, w1)

data_interval = np.linspace(-1.0, +1.0, 101)
linear_model = data_interval*w1+ w0

plt.figure(figsize = (10,6))
plt.plot(x_points, y_points, "b.", markersize = 5 )
plt.plot(data_interval, linear_model, "r")
plt.show()