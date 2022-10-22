#! usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

filename = "regression_points.csv" # Enter a file name  

data_set = np.genfromtxt(filename, delimiter= ",")

x_points = data_set[:,0]
y_points = data_set[:,1]

"""
plt.figure(figsize = (10, 6))
plt.plot(x_points , y_points, "r.", markersize = 5)
plt.show()
"""
# polynomial regression y_hat = w0 + w1*x + w2*x**2 + ... + wn*x**n
# Here I will assume that n = 2,
# To make this work N >= K 

N = x_points.shape[0] # N 
K = 2 # Enter the polynom you want

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

# lets find q_vector = [w0, w1, w2, .., wn]

q_vector = np.dot(np.linalg.inv(A_matrix), b_vector)


# Plotting the polynom


data_interval = np.linspace(-1.0, +1.0, 101) # Enter the interval and point size 
plynomial_model = np.stack([
        q_vector[c]*data_interval**c for c in range(q_vector.shape[0])
])

polynomial_model_T = plynomial_model.T

polynomial_model_final = np.array( [ np.sum(polynomial_model_T[c]) for c in range(polynomial_model_T.shape[0])]  )


plt.figure(figsize = (10,6))
plt.plot(x_points , y_points, "b.", markersize = 5)
plt.plot(data_interval , polynomial_model_final, "r")
plt.show()


