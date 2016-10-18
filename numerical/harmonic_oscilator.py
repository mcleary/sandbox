import ploter

__author__ = 'mcleary'

def main():    
    mass = 1.0 
    spring_constant = 2.0
    damping_constant = 0.0
    
    inv_mass = 1.0 / mass
    
    def dxdt(x, y): return y
    def dydt(x, y): return (-damping_constant * inv_mass * y) - (spring_constant * inv_mass * x)    
    
    ploter.plot_vector_field(dxdt, dydt, normalize=False, grid_resolution=20)
    

if __name__ == '__main__':
    main()