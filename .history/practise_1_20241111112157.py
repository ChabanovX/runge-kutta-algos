# This program solves first-order DEQ.
# For assignment to be completed I am to 
    # Implement Runge-Kutta Methods
    # Calculate and plot errors
    
    
import numpy as np
import matplotlib.pyplot as plt


def rk2(f, x0, xf,  step):
    x_vals = np.arange(x0, xf + step, step)
    print(x_vals)

rk2(1, 2, 2, )