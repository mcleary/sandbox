import numpy as np
import matplotlib.pyplot as plt
import math

__author__ = 'mcleary'


def example1():
    """
    This example show the simplest usage possible for matplotlib
    """
    plt.plot([1, 2, 3, 4])
    plt.ylabel('some numbers')
    plt.show()


def example2():
    """
    This example shows a simple line styling example
    """
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
    plt.axis([0, 6, 0, 20])
    plt.show()


def example3():
    """
    This example shows show to use matplotlib with numpy arrays
    """
    # evenly sampled time at 200ms interval
    t = np.arange(0.0, 5.0, 0.2)
    print(t)

    # red dashes, blue squares and green triangles
    plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
    plt.show()


def example4():
    """
    This example shows how to plot math vectors with matplotlib
    """
    soa = np.array([
        [0, 0, 3, 2],
        [0, 0, 1, 1],
        [0, 0, 9, 9]
    ])
    X, Y, U, V = zip(*soa)
    plt.figure()
    ax = plt.gca()
    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
    ax.set_xlim([-1, 10])
    ax.set_ylim([-1, 10])
    plt.draw()
    plt.show()


def example5():
    """
    This example shows how to plot a vector field in the hardest way
    """
    x_range = [-10.0, 10.0]
    y_range = [-10.0, 10.0]
    resolution = 1.0

    X = np.arange(x_range[0], x_range[1], resolution)
    Y = np.arange(y_range[0], y_range[1], resolution)

    # dxdt = y
    # dydt = -x
    #
    # Solution to this system is
    # x = a sin t
    # y = a cos t
    def dxdt(_x, _y): return _y
    def dydt(_x, _y): return -_x

    def normalize(_x, _y):
        length = math.sqrt(_x*_x + _y*_y)
        if length != 0.0 :
            return _x / length, _y / length
        else:
            return _x, _y,

    X1 = []
    Y1 = []
    U = []
    V = []
    for x in X:
        for y in Y:
            u = dxdt(x, y)
            v = dydt(x, y)
            u, v = normalize(u, v)
            X1.append(x)
            Y1.append(y)
            U.append(u)
            V.append(v)

    plt.figure()
    ax = plt.gca()
    ax.quiver(X1, Y1, U, V, angles='xy', scale_units='xy', scale=1)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    plt.draw()
    plt.show()


def example6():
    """
    This example shows how to plot a vector field in a much simpler manner
    """
    # generate grid
    print('Grid Resolution (10 is recomended): ', end='')
    grid_resolution = int(input())
    x = np.linspace(-10, 10, grid_resolution)
    y = np.linspace(-10, 10, grid_resolution)
    x, y = np.meshgrid(x, y)

    # Lorenz-attractor projected into z=0
    # calculate vector field
    # alpha = 3
    # rho = 26.5
    # vx = alpha*(y - x) / np.sqrt(x**2 + y**2)
    # vy = (rho*x - y) / np.sqrt(x**2 + y**2)        
    
    vx = y / np.sqrt(x**2 + y**2)    
    vy = -x / np.sqrt(x**2 + y**2)    

    # plot vector field
    plt.quiver(x, y, vx, vy, pivot='middle', headwidth=4, headlength=6)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.axis('image')
    plt.show()


def main():
    print('Choose example to run (1 to 6): ', end='')
    read_number = int(input())
    if read_number not in range(1, 7):
        print('Example {example_number} not found.'.format(example_number=read_number))
    else:
        print('Running example {example_number} ...'.format(example_number=read_number))
        globals()['example' + str(read_number)]()

if __name__ == '__main__':
    main()
