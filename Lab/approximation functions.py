import math
import pandas as pd
from itertools import zip_longest


def func(x, c=2):
    """Return value of function f(x)=x^2+6x+8-2e^(x+2) with c numbers after point"""
    return round(x ** 2 + 6 * x + 8 - 2 * math.exp(x + 2), c)


def finite_difference_coefficient(x, y):
    res = [x, y]
    while len(res[-1]) > 1:
        res.append([round(res[-1][i] - res[-1][i - 1], 2) for i in range(1, len(res[-1]))])
    return res


# interval for approximation
a, b = -6, 0
# index last node point
n = 3
node_point = [round(0.5 * (b + a) - (b - a) * 0.5 * math.cos(math.pi * (2 * i + 1) / (2 * n + 2)), 2) for i in range(n + 1)]
value_node_point = list(map(func, node_point))
finite_diff_coef = list(zip_longest(*finite_difference_coefficient(node_point, value_node_point)))
print(*finite_diff_coef, sep='\n')

