__author__ = 'mcleary'

import math

# f(x) = x^3 - 2x - 5
# dfdx = 3x^2 - 2


def f(x):
    # return (x * x * x) - (2 * x) - 5
    return math.pow(x, 6) - 2


def dfdx(x):
    # return (3 * x * x) - 2
    return 6 * math.pow(x, 5)


xn = 10.0
for i in xrange(100):
    xn1 = xn - (f(xn) / dfdx(xn))
    xn = xn1
    print(i, '{0:.10f}'.format(xn))
    pass
