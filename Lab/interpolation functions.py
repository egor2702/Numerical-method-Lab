import math
import pandas as pd
from itertools import zip_longest
import numpy as np
import matplotlib.pyplot as plt


def func(x, c=2):
    """Return value of function f(x)=x^2+6x+8-2e^(x+2) with c numbers after point"""
    return np.round(x ** 2 + 6 * x + 8 - 2 * np.exp(x + 2), c)


def finite_difference_coefficient(x, y):
    """Returns matrix of finite difference coefficient"""
    res = [x, y]
    while len(res[-1]) > 1:
        res.append([round(res[-1][i] - res[-1][i - 1], 2) for i in range(1, len(res[-1]))])
    return list(zip_longest(*res))


def lagrange_approximation(x):
    """Returns value of founded Lagrange's polynomial"""
    return -0.189471 * x ** 3 - 1.43203 * x ** 2 - 4.13458 * x - 5.9475


def newton_approximation(x):
    """Returns value of founded Newton's polynomial"""
    return -6.78 - 4.74417 * x - 1.575 * x ** 2 - 0.198958 * x ** 3


def spline(x):
    """Gives vector x and return value of spline interpolation for each coordinate"""
    result_list = []
    for i in x:
        if i > -2:
            result_list.append(-6.78 - 2.771 * i + 0.095 * i ** 3)
        else:
            result_list.append(-8.304 - 5.056 * i - 1.142 * i ** 2 - 0.095 * i ** 3)
    return np.array(result_list)


# interval for approximation
a, b = -6, 0
# index last node point
n = 3
# node_point1 = [round(0.5 * (b + a) - (b - a) * 0.5 * math.cos(math.pi * (2 * i + 1) / (2 * n + 2)), 2) for i in
#                range(n + 1)]
# value_node_point1 = list(map(func, node_point1))
# finite_diff_coef1 = finite_difference_coefficient(node_point1, value_node_point1)
# print(*finite_diff_coef1, sep='\n')
#
# node_point2 = [-6, -4, -2, 0]
# value_node_point2 = list(map(func, node_point2))
# finite_diff_coef2 = finite_difference_coefficient(node_point2, value_node_point2)
# print(*finite_diff_coef2, sep='\n')

check_points = [-5, -3, -1.5, -2.1]
# print(list(map(func, check_points)))
# print(list(map(lagrange_approximation, check_points)))
# print(list(map(newton_approximation, check_points)))

x = np.arange(-6, 0, 0.01)

fig_lagrange = plt.figure()
ax1 = fig_lagrange.add_subplot()
ax1.plot(x, lagrange_approximation(x), color='r', label='Lagrange')
ax1.plot(x, func(x), '--', color='g', label='function', linewidth=3)
ax1.grid()
ax1.legend()

fig_newton = plt.figure()
ax2 = fig_newton.add_subplot()
ax2.plot(x, newton_approximation(x), 'r', label='Newton')
ax2.plot(x, func(x), '--', color='g', label='function', linewidth=3)
ax2.grid()
ax2.legend()

fig_spline = plt.figure()
ax3 = fig_spline.add_subplot()
ax3.plot(x, spline(x), 'r', label='Spline')
ax3.plot(x, func(x), '--', color='g', label='function', linewidth=3)
ax3.grid()
ax3.legend()
plt.show()
