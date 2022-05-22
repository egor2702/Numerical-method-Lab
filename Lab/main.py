import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as AA
from matplotlib.ticker import MultipleLocator
from numpy import linalg
from scipy.optimize import fsolve

EPS = 1e-5
phi_1 = lambda x: (np.sin(x[1]) - 1) / 2
phi_2 = lambda x: np.cos(x[0] + 0.5) - 2

func_tup = (phi_1, phi_2)
x_0 = np.array([-1, -1])


def paint_plot_1(a, b):
    """Create plot with two functions"""
    y1 = np.arange(a, b, 0.1)
    x1 = (np.sin(y1) - 1) / 2
    x2 = np.arange(a, b, 0.1)
    y2 = np.cos(x2 + 0.5) - 2
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(axes_class=AA.Axes)
    ax.axis["right"].set_visible(False)  # hide right axis
    ax.axis["left"].set_visible(False)
    ax.axis["top"].set_visible(False)
    ax.axis["bottom"].set_visible(False)
    ax.axis["y=0"] = ax.new_floating_axis(nth_coord=0, value=0)  # Add new axis which will be centred
    ax.axis["x=0"] = ax.new_floating_axis(nth_coord=1, value=0)
    ax.set(xlim=(-6, 6), ylim=(-6, 6))
    graph1 = plt.plot(x1, y1, linewidth=3, color=(0, 0, 1))
    graph2 = plt.plot(x2, y2, linewidth=3, color=(252 / 256, 227 / 256, 0))
    ax.grid()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    plt.show()


def fixed_point_iteration(x_k0, func_tup, eps):
    """Realization of method Fixed point iteration"""
    i = 0
    print(f"{'iteration':^11s}" + ' ' * 5 + 'x' + ' ' * 9 + 'y' + ' ' * 8 + "delta")
    print(f"{i:^11d} {x_k0[0]:^7.6f} {x_k0[1]:^7.6f}")
    while True:
        x_k1 = np.array([func_tup[i](x_k0) for i in range(x_k0.size)])
        i += 1
        print(f"{i:^11d} {x_k1[0]:^7.6f} {x_k1[1]:^7.6f}  {linalg.norm(x_k0 - x_k1, np.inf):7.6f}")
        if linalg.norm(x_k0 - x_k1, np.inf) <= eps:
            return x_k1
        x_k0 = np.copy(x_k1)


def func(x):
    """Take (x, y) vector and return vector (cos(x + 0.5) - y - 2; sin(y) - 2x - 1)"""
    return [np.cos(x[0] + 0.5) - x[1] - 2, np.sin(x[1]) - 2 * x[0] - 1]


# paint_plot_1(-6, 6)
# # Solve with first approximate (-1,-1)
# solution1 = fixed_point_iteration(x_0, func_tup, EPS)
# print('solution: ', *solution1.round(5))
# # Solve with random first approximate
# solution2 = fixed_point_iteration((np.random.rand(1, 2) * 100).reshape(-1), func_tup, EPS)
# print('solution: ', *solution2.round(5))
# # Use function from scipy.optimize for solving equation
# root = fsolve(func, x_0)
# print('solution from library function fsolve', *root.round(5))


def paint_plot_2():
    """Create plot with two functions"""
    x1 = np.arange(-2, 2, 0.01)
    y1 = np.arctan(np.power(x1, 2)) / x1
    x2 = np.zeros([1, 3], dtype=np.int16).reshape(-1)
    y2 = np.linspace(-3, 3, 3)
    x3 = np.linspace(-1 / (0.7 ** 0.5), 1 / (0.7 ** 0.5), 200)
    y3 = np.power((1 - 0.7 * np.power(x3, 2)) / 2, 0.5)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(axes_class=AA.Axes)
    ax.axis["right"].set_visible(False)  # hide right axis
    ax.axis["left"].set_visible(False)
    ax.axis["top"].set_visible(False)
    ax.axis["bottom"].set_visible(False)
    ax.axis["y=0"] = ax.new_floating_axis(nth_coord=0, value=0)  # Add new axis which will be centred
    ax.axis["x=0"] = ax.new_floating_axis(nth_coord=1, value=0)
    ax.set(xlim=(-2, 2), ylim=(-2, 2))
    plt.plot(x1, y1, x2, y2, linewidth=3, color=(0, 0, 1))
    plt.plot(x3, y3, x3, -y3, linewidth=3, color=(252 / 256, 227 / 256, 0))
    ax.grid()
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    plt.show()


def func_1(x_vect):
    """"Take vector (x,y) and return (tan(xy) - x^2; 0.7x^2 + 2y^2 - 1)"""
    x = np.tan(x_vect[0] * x_vect[1]) - x_vect[0] ** 2
    y = 0.7 * x_vect[0] ** 2 + 2 * x_vect[1] ** 2 - 1
    return np.array([x, y])


def yakobi_mat(x_vect):
    """Take vector (x, y) and return matrix of coefficient W """
    a11 = x_vect[1] / (np.cos(x_vect[0] * x_vect[1]) ** 2) - 2 * x_vect[0]
    a12 = x_vect[0] / (np.cos(x_vect[0] * x_vect[1]) ** 2)
    a21 = 1.4 * x_vect[0]
    a22 = 4 * x_vect[1]
    return np.array([[a11, a12], [a21, a22]])


def newton_method(x_k0, w_mat, f_vect, eps):
    i = 0
    print(f"{'iteration':^11s}" + ' ' * 5 + 'x' + ' ' * 9 + 'y' + ' ' * 8 + "delta")
    print(f"{i:^11d} {x_k0[0]:^9.6f} {x_k0[1]:^9.6f}")
    while True:
        d_x = np.linalg.solve(w_mat(x_k0), -f_vect(x_k0))
        x_k1 = x_k0 + d_x
        print(f"{i:^11d} {x_k1[0]:^9.6f} {x_k1[1]:^9.6f}  {linalg.norm(x_k0 - x_k1, np.inf):8.6f}")
        if np.linalg.norm(x_k1 - x_k0, np.inf) < eps:
            return x_k1
        i += 1
        x_k0 = x_k1


# paint_plot_2()
first_approximate = ((0, 0.7), (0, -0.7), (0.6, 0.6), (-0.6, -0.6))
# Solve with first approximation (0, 0.7), (0, -0.7), (0.6, 0.6), (-0.6, -0.6)
for i in range(4):
    solution = newton_method(np.array(first_approximate[i]), yakobi_mat, func_1, EPS).round(5)
    print(f"solution: x = {solution[0]}  y = {solution[1]}")
    print(f"F(x_*) = {func_1(solution)}\n")
# Solve with random first approximation
print("Solution: ", newton_method((np.random.rand(1, 2)*100).reshape(-1), yakobi_mat, func_1, EPS).round(5))
# Use function from scipy.optimize for solving equation
for i in range(4):
    root = fsolve(func_1, first_approximate[i])
    print('solution from library function fsolve: ', *root.round(5))
