import numpy as np
import matplotlib.pyplot as plt

def plot_vector_field(dxdt, dydt, x_range=[-100,100], y_range=[-100,100], grid_resolution=10, normalize=True):    
    """    
    Plots a vector field with vectors generated from dxdt and dydt functions.
    :param dxdt: Function of (x,y) that returns the x-component of a vectors with origin in (x,y)
    :param dydt: Function of (x,y) that returns the y-component of a vectors with origin in (x,y)
    :param x_range limits of the grid in x-axis
    :param y_range limits of the grid in y-axis
    :param grid_resolution: number of points to generate in the grid for each axis
    :normalize: Should the vector field be normalized? In this case it becaomes a direction field.
    """
    
    # Generate Mesh Grid
    x = np.linspace(x_range[0], x_range[1], grid_resolution)
    y = np.linspace(y_range[0], y_range[1], grid_resolution)
    x, y = np.meshgrid(x, y)
    
    # Generate Vector Field
    vx = dxdt(x,y)
    vy = dydt(x,y)    
    
    if normalize:
        norm = 1 / np.sqrt(vx**2 + vy**2)
        vx = vx * norm
        vy = vy * norm
    
    # plot vector field
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.quiver(x, y, vx, vy, pivot='middle', headwidth=4, headlength=6)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.axis('image')
    plt.show()