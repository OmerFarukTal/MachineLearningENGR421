#! usr/bin/env/ python3

import numpy as np
from scipy.stats import multivariate_normal



x = np.array([  [1,1.2],
                [1.2, 3]])

y  = np.mean(x, axis = 0)
print(y)

k = np.array([1,2])

z = np.array([[1,2],
                [2,4]])

meraba = multivariate_normal.pdf(z,y,x)
print(meraba)