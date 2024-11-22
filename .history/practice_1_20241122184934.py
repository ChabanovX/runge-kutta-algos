# First task solution
# IVAN CHABANOV

import numpy as np
import matplotlib.pyplot as plt


def rk2(f, x0, xf, y0, step):
    x_vals = np.arange(x0, xf + step, step)
    y_vals = np.zeros(len(x_vals))
    y_vals[0] = y0
    
    for i in range(1, len(x_vals)):
        x_n = x_vals[i - 1]
        y_n = y_vals[i - 1]
        k1 = f(x_n, y_n)
        k2 = f(x_n + step,
               y_n + step * k1)
        
        y_vals[i] = y_n + step * (k1 + k2) / 2 
    
    return x_vals, y_vals


def rk4(f, x0, xf, y0, step):
    x_vals = np.arange(x0, xf + step, step)
    y_vals = np.zeros(len(x_vals))
    y_vals[0] = y0
    
    for i in range(1, len(x_vals)):
        x_n = x_vals[i - 1]
        y_n = y_vals[i - 1]
        k1 = f(x_n, y_n)
        k2 = f(x_n + step / 2,
               y_n + step / 2 * k1)
        k3 = f(x_n + step / 2, 
               y_n + step / 2 * k2)
        k4 = f(x_n + step,
               y_n + step * k3)
        
        y_vals[i] = y_n + step * (k1 + 2*k2 + 2*k3 + k4) / 6

    return x_vals, y_vals


def f1(x, y):
    return x + np.cos(y)


def f2(x, y):
    return x**2 + y**2


def main():
    equations = {'a': (f1, 1, 30), 'b': (f2, 2, 1)}
    methods = {'RK2': rk2, 'RK4': rk4}
    hs = [0.1, 0.05, 0.01, 0.005, 0.001]
    N
    
    pass


# Initial conditions:

print(rk2(f1, 1, 2, 1, 0.1))
print()
print(rk4(f1, 1, 2, 1, 0.1))
