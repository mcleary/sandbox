import math
import sys




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
    for i_iteration in range(max_iterations):
        next_approx = previous_approx - (function(previous_approx) / derivative(previous_approx))
        numeric_error = math.fabs(next_approx - previous_approx)
        previous_approx = next_approx

        if verbose:
            print(('Iteration {iteration}. Approx: {approx:.8f}. Error: {error:.8f}'.format(iteration=i_iteration, approx=next_approx, error=numeric_error)))

        if numeric_error < sys.float_info.epsilon:
            break

    return next_approx


if __name__ == '__main__':
    # f(x) = x^3 - 2x - 5
    # dfdx = 3x^2 - 2

    def f(x): return (x * x * x) - (2 * x) - 5

    def dfdx(x): return (3 * x * x) - 2

    root = find_root(function=f, derivative=dfdx, first_approximation=2, verbose=True)
    print(('The root of x^3 - 2x - 5 = 0 is {0:.30f}'.format(root)))
    f_root = f(root)
    print(('Replacing the root value in the function gives: {0}, which is preety close to 0.'.format(f_root)))

