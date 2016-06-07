import numpy as np

def main():
    # x = np.array([0, 0.25, 0.5, 0.75, 1])
    # y = np.array([1, 1.2840, 1.6487, 2.1170, 2.7183])

    x = [0, 1, 2, 3, 4, 5, 6]
    y = [32, 47, 65, 92, 132, 190, 275]

    m = len(x)

    sum_x = x.sum()
    sum_x2 = 0
    for xi in x:
        sum_x2 += xi**2

    sum_y = y.sum()
    sum_xy = x.dot(y)

    A = [[m, sum_x], [sum_x, sum_x2]]
    b = [sum_y, sum_xy]

    print np.linalg.solve(A, b)




if __name__ == '__main__':
    main()