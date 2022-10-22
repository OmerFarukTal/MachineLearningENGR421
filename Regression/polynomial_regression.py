#! usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(421)

x_points = np.sort(np.random.random(100))
x_s = np.random.random(100)
y_points = np.sort(x_s**2*np.random.random(1) + x_s*np.random.random(1) + np.random.random(1))

np.savetxt("regression_points.csv", np.stack((x_points, y_points), axis=1), fmt="%f,%f")

"""
plt.figure(figsize = (10, 6))
plt.plot(x_points , y_points, "r.", markersize = 5)
plt.show()
"""

"""
We created the data points
We can save them to somerwhere 
In the rest of the code I will assume that I took this data from somewhere else 
"""

# polynomial regression y_hat = w0 + w1*x + w2*x**2 + ... + wn*x**n
# Here I will assume that n = 2,
# To make this work N >= K 

N = x_points.shape[0]
K = 2

# lets find q_vector which entries are q_vector = [w0, w1, w2, .., wn]
# Aq = b
# D*D.T = A
# q = A**-1*b

D_matrix = np.stack( [
        x_points**c for c in range(K + 1)
    ] 
).T


A_matrix = np.dot(D_matrix.T, D_matrix)


b_vector = np.stack([
        np.dot(y_points.T, x_points**c) for c in range(K+1)
    ]
)

# lets find q_vector 

q_vector = np.dot(np.linalg.inv(A_matrix), b_vector)

print(q_vector)

data_interval = np.linspace(-1.0, +1.0, 101)
plynomial_model = np.stack([
        q_vector[c]*data_interval**c for c in range(q_vector.shape[0])
])

polynomial_model_T = plynomial_model.T

polynomial_model_final = np.array( [ np.sum(polynomial_model_T[c]) for c in range(polynomial_model_T.shape[0])]  )


plt.figure(figsize = (10,6))
plt.plot(x_points , y_points, "b.", markersize = 5)
plt.plot(data_interval , polynomial_model_final, "r")
plt.show()

