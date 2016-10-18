import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings('ignore')

global_tol = 10e-2
start_point = [100, 100]
graph_range = [-200, 200]


def f(x):
    return x[0] ** 2 + 10 * x[1] ** 2


def grad_f(x):
    return np.array([2 * x[0], 20 * x[1]])


def grad_descent_fss(x0, max_iter=50):
    if np.linalg.norm(grad_f(x0)) < global_tol:
        return []

    x_seq = []

    iter = 0
    xk = x0
    while iter < max_iter and np.linalg.norm(grad_f(xk)) > global_tol:
        dk = -grad_f(xk)
        lk = 0.1
        x_seq.append(np.copy(xk))
        xk = xk + (lk * dk)
        iter += 1

    return np.array(x_seq)


def grad_descent_btls(x0, max_iter=20):
    if np.linalg.norm(grad_f(x0)) < global_tol:
        return []

    x_seq = []

    iter = 0
    xk = x0
    while iter < max_iter and np.linalg.norm(grad_f(xk)) > global_tol:
        dk = -grad_f(xk)

        # Backtrack line search
        lk = 1.0
        alpha = 0.1
        beta = 0.3
        while f(xk + lk * dk) > (f(xk) + alpha * lk * dk.dot(dk)):
            lk *= beta

        x_seq.append(np.copy(xk))
        xk += lk * dk

        iter += 1

    return np.array(x_seq)


def main():
    #seq_x = grad_descent_fss(start_point, max_iter=20)
    seq_x = grad_descent_btls(start_point)
    for s in seq_x:
        print(s)
    X = np.arange(graph_range[0], graph_range[1], 1)
    Y = np.arange(graph_range[0], graph_range[1], 1)
    X, Y = np.meshgrid(X, Y)
    Z = f([X, Y])

    plt.figure()
    plt.contour(X, Y, Z)
    plt.plot(seq_x[:, 0], seq_x[:, 1])
    plt.show()


main()