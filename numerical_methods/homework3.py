# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

x = [0.9, 1.3, 1.9, 2.1, 2.6, 3.0, 3.9, 4.4, 4.7, 5.0, 6.0, 7.0, 8.0, 9.2, 10.5, 11.3, 11.6, 12.0, 12.6, 13.0, 13.3]
y = [1.3, 1.5, 1.85, 2.1, 2.6, 2.7, 2.4, 2.15, 2.05, 2.1, 2.25, 2.3, 2.25, 1.95, 1.4, 0.9, 0.7, 0.6, 0.5, 0.4, 0.25]


def main():
    n = len(x)
    A = [[0 for _ in range(3*(n-1))] for _ in range(3*(n-1))]
    b = [0 for _ in range(3*(n-1))]

    row_idx = 0
    for i in range(1, n):
        col_idx = (i-1)*3

        # a[i] * x[i-1]**2 + b[i] * x[i-1] + c[i] = y[i-1]
        b[row_idx] = y[i-1]
        A[row_idx][col_idx + 0] = x[i-1]**2
        A[row_idx][col_idx + 1] = x[i-1]
        A[row_idx][col_idx + 2] = 1.0

        row_idx += 1

        # a[i] * x[i]**2 + b[i] * x[i] + c[i] = y[i]
        b[row_idx] = y[i]
        A[row_idx][col_idx + 0] = x[i]**2
        A[row_idx][col_idx + 1] = x[i]
        A[row_idx][col_idx + 2] = 1.0

        row_idx += 1

    for i in range(1, n-1):
        col_idx = (i-1)*3

        # 2a[i] * x[i] + b[i] - 2a[i+1] * x[i] - b[i+1] = 0
        A[row_idx][col_idx + 0] = 2.0 * x[i]
        A[row_idx][col_idx + 1] = 1.0
        # A[row_idx][col_idx + 2] = 0.0
        A[row_idx][col_idx + 3] = -2.0 * x[i]
        A[row_idx][col_idx + 4] = -1.0

        row_idx += 1

    # last equation: a0 = 0
    # A[row_idx][3*(n-1)-1] = 1
    A[row_idx][0] = 1
    b[row_idx] = 0

    coef = np.linalg.solve(A, b)

    xarr = np.asarray(x)
    xnew = np.arange(xarr.min(), xarr.max(), 0.01)
    ynew = np.zeros(len(xnew))

    for i in range(len(xnew)):
        curr_x = xnew[i]
        for j in range(len(x)-1):
            if (curr_x >= x[j]) and (curr_x <= x[j+1]):
                aa = coef[j*3 + 0]
                bb = coef[j*3 + 1]
                cc = coef[j*3 + 2]
                ynew[i] = aa * curr_x**2 + bb * curr_x + cc
                break

    plt.figure()
    plt.plot(x, y, 'sr', xnew, ynew, 'b')
    plt.axis([xarr.min(), xarr.max(), ynew.min(), ynew.max()])
    plt.show()


if __name__ == '__main__':
    main()