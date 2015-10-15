__author__ = 'mcleary'


def euler_solver(equation, x0, y0, step_size=0.01, max_iterations=100, verbose=False):
    x = float(x0)
    y = float(y0)
    xn = [x]
    yn = [y]
    for i_iteration in xrange(max_iterations):
        previous_y = yn[i_iteration]
        previous_x = xn[i_iteration]
        next_x = previous_x + step_size
        next_y = previous_y + step_size * equation(previous_x, previous_y)
        xn.append(next_x)
        yn.append(next_y)

        if verbose:
            print('n: {iteration} \t x{iteration}: {xn:.8f} \t y{iteration}: {yn:.8f}'.format(iteration=i_iteration+1,
                                                                                              xn=xn[i_iteration + 1],
                                                                                              yn=yn[i_iteration + 1]))
    return xn, yn


def main():
    def f(x, y): return x + y

    xn, yn = euler_solver(equation=f, x0=0.0, y0=1.0, verbose=True)

    print xn, yn

if __name__ == '__main__':
    main()

