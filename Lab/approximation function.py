import numpy as np
from matplotlib import pyplot as plt

x = np.array([2.00, 2.22, 2.44, 2.67, 2.89, 3.11, 3.33, 3.56, 3.78, 4.00])
y = np.array([1.52, 1.84, 1.68, 1.34, 1.67, 1.35, 1.44, 1.43, 0.91, 1.09])

F_t = np.array([[1] * 10, [i for i in x], [i ** 2 for i in x]])
F = F_t.transpose()
A = F_t @ F
coef_vect = np.linalg.inv(A) @ F_t @ y.transpose()


def approximation_func(x):
    return coef_vect[0] + coef_vect[1] * x + coef_vect[2] * x ** 2


fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(x, y)
ax.plot(np.arange(0, 5, 0.1), approximation_func(np.arange(0, 5, 0.1)), 'r')
ax.grid()
plt.axis([0, 5, 0, 2])
plt.show()

print(coef_vect)
