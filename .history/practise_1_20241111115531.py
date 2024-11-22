# This program solves first-order DEQ.
# For assignment to be completed I am to 
    # Implement Runge-Kutta Methods
    # Calculate and plot errors
    
    
import numpy as np
import matplotlib.pyplot as plt


def f1(x, y):
    return x + np.cos(y)


def f2(x, y):
    return x**2 + y**2


def rk2(f, x0, xf, y0, step):
    x_vals = np.arange(x0, xf + step, step)
    y_vals = np.zeros(len(x_vals))
    y_vals[0] = y0
    
    for i in range(1, len(x_vals)):
        x_n = x_vals[i - 1]
        y_n = y_vals[i - 1]
        
        k1 = f(x_n, y_n)
        k2 = f(x_n + step, y_n + step * k1)
        y_vals[i] = y_n + step * (k1 + k2) / 2 
    
    return x_vals, 

rk2(f1, 1, 2, 1, 0.1)
