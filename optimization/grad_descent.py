import numpy as np


def grad_descent_fss(x0, grad_f, max_iter=20, tol=10e-2, step_size=0.1):
    if np.linalg.norm(grad_f(x0)) < tol:
        return []

    x_seq = []

    iter = 0
    xk = x0
    while iter < max_iter and np.linalg.norm(grad_f(xk)) > tol:
        dk = -grad_f(xk)
        x_seq.append(xk)
        xk += step_size * dk
        iter += 1

    return np.array(x_seq)


def grad_descent_btls(x0,  f, grad_f, max_iter=20, tol=10e-2, alpha=0.1, beta=0.3):
    if np.linalg.norm(grad_f(x0)) < tol:
        return []

    x_seq = []

    iter = 0
    xk = x0
    while iter < max_iter and np.linalg.norm(grad_f(xk)) > tol:
        dk = -grad_f(xk)

        # Backtrack line search
        step_size = 1.0
        while f(xk + step_size * dk) > (f(xk) + alpha * step_size * dk.dot(dk)):
            step_size *= beta

        x_seq.append(np.copy(xk))
        xk += step_size * dk

        iter += 1

    return np.array(x_seq)
