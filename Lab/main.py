import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as AA
from matplotlib.ticker import MultipleLocator
from numpy import linalg


EPS = 1e-5
phi_1 = lambda x: (np.sin(x[1]) - 1) / 2
phi_2 = lambda x: np.cos(x[0] + 0.5) - 2

func_tup = (phi_1, phi_2)
x_0 = np.array([-1, -1])


def paint_plot(a, b):
    """Create plot with two functions"""
    y1 = np.arange(a, b, 0.1)
    x1 = (np.sin(y1) - 1) / 2
    x2 = np.arange(a, b, 0.1)
    y2 = np.cos(x2 + 0.5) - 2
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(axes_class=AA.Axes)
    ax.axis["right"].set_visible(False)
    ax.axis["left"].set_visible(False)
    ax.axis["top"].set_visible(False)
    ax.axis["bottom"].set_visible(False)
    ax.axis["y=0"] = ax.new_floating_axis(nth_coord=0, value=0)
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


solution1 = fixed_point_iteration(x_0, func_tup, EPS)
print('solution: ', *solution1.round(5))

solution2 = fixed_point_iteration((np.random.rand(1, 2)*100).reshape(-1), func_tup, EPS)
print('solution: ', *solution2.round(5))


