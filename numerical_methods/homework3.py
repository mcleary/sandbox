# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

x = [0.9, 1.3, 1.9, 2.1, 2.6, 3.0, 3.9, 4.4, 4.7, 5.0, 6.0, 7.0, 8.0, 9.2, 10.5, 11.3, 11.6, 12.0, 12.6, 13.0, 13.3]
y = [1.3, 1.5, 1.85, 2.1, 2.6, 2.7, 2.4, 2.15, 2.05, 2.1, 2.25, 2.3, 2.25, 1.95, 1.4, 0.9, 0.7, 0.6, 0.5, 0.4, 0.25]

# x = [0, 10, 15, 20, 22.5, 30]
# y = [0, 227.04, 362.78, 517.35, 602.97, 901.67]


def numpy_main():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import interpolate

    tck = interpolate.splrep(x, y, s=0, k=2)
    xnew = np.arange(0, 15, 0.2)
    ynew = interpolate.splev(xnew, tck, der=0)

    plt.figure()
    plt.plot(x, y, 'x', xnew, ynew, 'b')
    plt.axis([0, 15, 0, 5])
    plt.show()


def main():
    n = len(x)
    coef = [[0 for _ in range(3*(n-1))] for _ in range(3*(n-1))]
    b = [0 for _ in range(3*(n-1))]

    row_idx = 0
    for i in range(1, n):
        col_idx = (i-1)*3

        # a[i] * x[i-1]**2 + b[i] * x[i-1] + c[i] = y[i-1]
        b[row_idx] = y[i-1]
        coef[row_idx][col_idx + 0] = x[i-1]**2
        coef[row_idx][col_idx + 1] = x[i-1]
        coef[row_idx][col_idx + 2] = 1.0

        row_idx += 1

        # a[i] * x[i]**2 + b[i] * x[i] + c[i] = y[i]
        b[row_idx] = y[i]
        coef[row_idx][col_idx + 0] = x[i]**2
        coef[row_idx][col_idx + 1] = x[i]
        coef[row_idx][col_idx + 2] = 1.0

        row_idx += 1

    for i in range(1, n-1):
        col_idx = (i-1)*3

        # 2a[i] * x[i] + b[i] - 2a[i+1] * x[i] - b[i+1] = 0
        coef[row_idx][col_idx + 0] = 2.0 * x[i]
        coef[row_idx][col_idx + 1] = 1.0
        coef[row_idx][col_idx + 3] = -2.0 * x[i]
        coef[row_idx][col_idx + 4] = -1.0

        row_idx += 1

    # last equation: a1 = 0
    coef[row_idx][0] = 1.0
    b[row_idx] = 0.0

    # coef[row_idx][0] = 2.0 * x[0]
    # coef[row_idx][1] = x[0]
    # b[row_idx] = 0

    scoef = np.linalg.solve(coef, b)

    xarr = np.asarray(x)
    xnew = np.arange(xarr.min(), xarr.max(), 0.01)
    ynew = np.zeros(len(xnew))

    scoef[0] = 0.0

    for i in range(len(xnew)):
        curr_x = xnew[i]
        for j in range(len(x)-1):
            if (curr_x >= x[j]) and (curr_x <= x[j+1]):
                aa = scoef[j*3 + 0]
                bb = scoef[j*3 + 1]
                cc = scoef[j*3 + 2]
                ynew[i] = aa * curr_x**2 + bb * curr_x + cc


    plt.figure()
    plt.plot(x, y, 'x', xnew, ynew, 'b')
    #plt.plot(x, y, 'x')
    plt.axis([xarr.min(), xarr.max(), ynew.min(), ynew.max()])
    plt.show()



if __name__ == '__main__':
    # numpy_main()
    main()