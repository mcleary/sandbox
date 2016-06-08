import numpy as np
import math
import matplotlib.pyplot as plt

def mmq_1():
    x = np.array([0, 1, 2, 3, 4, 5, 6])
    y = np.array([32, 47, 65, 92, 132, 190, 275])

    ln_y = np.log(y)

    m = len(x)

    sum_x = x.sum()
    sum_x2 = 0
    for xi in x:
        sum_x2 += xi**2

    sum_y = ln_y.sum()
    sum_xy = x.dot(ln_y)

    A = [[m, sum_x], [sum_x, sum_x2]]
    b = [sum_y, sum_xy]

    z = np.linalg.solve(A, b)
    print A, b, z

    print 'g(x) = ' + str(math.exp(z[0])) + ' + exp(' + str(math.exp(z[1])) + 'x)'

    a = math.exp(z[0])
    b = z[1]
    def gx(x):
        return a * math.exp(b * x)

    t1 = (math.log(2000) - math.log(a)) / b
    print 'Tempo necessario para a populacao ultrapassar 2000: ' + str(t1)

    x_new = np.arange(x.min(), x.max(), 0.01)
    y_new = np.zeros(len(x_new))

    for i in range(len(x_new)):
        y_new[i] = gx(x_new[i])

    plt.figure()
    plt.plot(x, y, 'sr', x_new, y_new, 'b')
    plt.axis([x.min(), x.max(), y.min(), y.max()])
    plt.show()


def mmq_2():
    x = np.array([0, 1, 2, 3])
    fi = np.array([-0.2, 0.4, 0.2, 0.6])
    gi = np.array([1.2, 1.6, 1.8, 2])

    m = len(x)
    sum_x = x.sum()
    sum_x2 = 0
    for xi in x:
        sum_x2 += xi*xi
    sum_fi = fi.sum()
    sum_gi = gi.sum()
    sum_xfi = x.dot(fi)
    sum_xgi = x.dot(gi)

    A = [[m,     sum_x,  0],
         [sum_x, sum_x2, 0],
         [0,     sum_x,  m]]
    b = [sum_fi, sum_xfi, sum_gi]

    # A = [[m,     sum_x,  0],
    #      [sum_x, sum_x2, 0],
    #      [0,     sum_x2,  sum_x]]
    # b = [sum_fi, sum_xfi, sum_xgi]

    z = np.linalg.solve(A, b)
    print A, b, z

    a, b, c, d = z[0], z[1], z[2], z[1]

    print a, b, c, d

    def fx(x):
        return a + b*x

    def gx(x):
        return c + d*x

    x_new = np.arange(x.min(), x.max(), 0.01)
    y1_new = np.zeros(len(x_new))
    y2_new = np.zeros(len(x_new))
    for i in range(len(x_new)):
        y1_new[i] = fx(x_new[i])
        y2_new[i] = gx(x_new[i])

    plt.figure()
    plt.plot(x, fi, 'sr', x, gi, 'sb', x_new, y1_new, 'g', x_new, y2_new, 'c')
    plt.axis([x.min(), x.max(), fi.min(), fi.max()])
    plt.show()

if __name__ == '__main__':
    # mmq_1()
    mmq_2()