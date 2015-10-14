import math
import sys

__author__ = 'mcleary'

# f(x) = x^3 - 2x - 5
# dfdx = 3x^2 - 2


def find_root(function, derivative, first_approximation, max_iterations=100, verbose=False):
    """
    Calculates the root of a function f using the Newthon-Raphson iterative method
    :param function: function to find a root
    :param derivative: derivative of function f
    :param max_iterations: maximum number of iterations to run the method
    :param first_approximation: initial approximation to start the Newthon-Raphson method
    :return: a root of function f
    """
    previous_approx = float(first_approximation)
    for iIteration in xrange(max_iterations):
        next_approx = previous_approx - (function(previous_approx) / derivative(previous_approx))
        numeric_error = math.fabs(next_approx - previous_approx)
        previous_approx = next_approx

        if verbose:
            print(iIteration, 'Cur. Approx: ', next_approx, 'Error: ', numeric_error)

        if numeric_error < sys.float_info.epsilon:
            break

    return next_approx


if __name__ == '__main__':
    #root = find_root(function=f, derivative=dfdx, max_iterations=20, first_approximation=2.0, verbose=True)
    root = find_root(function=lambda x: (x * x * x) - (2 * x) - 5, derivative=lambda x: (3 * x * x) - 2,
                     first_approximation=2, verbose=True)
    print('{0:.10f}'.format(root))
