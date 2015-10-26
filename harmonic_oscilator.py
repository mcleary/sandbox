import numpy as np
import matplotlib.pyplot as plt


def main():
    print('Grid Resolution (10 is recomended): ', end='')
    grid_resolution = int(input())
    x = np.linspace(-100, 100, grid_resolution)
    y = np.linspace(-100, 100, grid_resolution)
    x, y = np.meshgrid(x, y)
    
    m = 1.5
    b = 0.5
    k = 0.5
    
    inv_m = 1.0 / m
    
    vx = y
    vy = (-b*inv_m*y - k*inv_m*x)
    
    vx = vx / np.sqrt(vx**2 + vy**2)
    vy = vy / np.sqrt(vx**2 + vy**2)

    # plot vector field
    plt.quiver(x, y, vx, vy, pivot='middle', headwidth=4, headlength=6)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.axis('image')
    plt.show()


if __name__ == '__main__':
    main()