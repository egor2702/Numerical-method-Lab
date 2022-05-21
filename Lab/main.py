import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as AA
from matplotlib.ticker import MultipleLocator

phi_1 = lambda y: (np.sin(y) - 1) / 2
phi_2 = lambda x: np.cos(x + 0.5) - 2


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


paint_plot(-5, 5)
