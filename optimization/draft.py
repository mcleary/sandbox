from grad_descent import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math


def f(x):
    return x[0]**2 + 10*x[1]**2


def grad_f(x):
    return np.array([2*x[0], 20*x[1]])


def h(x):
    return x[0]**2 + x[1]**2


def grad_h(x):
    return np.array([2*x[0], 2*x[1]])


def g(x):
    x0 = x[0]
    x1 = x[1]
    t1 = np.exp(x0 + 3*x1 - 0.1)
    t2 = np.exp(x0 - 3*x1 - 0.1)
    t3 = np.exp(-x0 - 0.1)
    return t1 + t2 + t3


def grad_g(x):
    x0 = x[0]
    x1 = x[1]
    c1 = 0.904837
    c2 = 2.71451
    t1 = c1 * np.exp(x0 - 3*x1) + c1 * np.exp(x0 + 3*x1) - c1 * np.exp(-x0)
    t2 = c2 * (np.exp(6*x1) - 1.0) * np.exp(x0 - 3*x1)
    return np.array([t1, t2])


start_point = [100, 100]
graph_res = 1

X = np.arange(-150, 150, graph_res)
Y = np.arange(-150, 150, graph_res)
X, Y = np.meshgrid(X, Y)
Z = f([X, Y])

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z)
# plt.show()

seq_1 = grad_descent_btls(start_point, f=f, grad_f=grad_f, max_iter=50, alpha=0.1, beta=0.7)
# seq_1 = grad_descent_fss(start_point, grad_f=grad_f)
seq_2 = grad_descent_acc(start_point, grad_f=grad_f)

plt.figure()
plt.contour(X, Y, Z)
# plt.xlim([-1.5, 1.5])
# plt.ylim([-0.4, 0.4])
plt.plot(seq_1[:, 0], seq_1[:, 1])
plt.plot(seq_2[:, 0], seq_2[:, 1])
plt.show()

print(seq_2)
