import numpy as np
import matplotlib.pyplot as plt

# Define the differential equations
def diff_eq1(x, y):
    return x + np.cos(y)

def diff_eq2(x, y):
    return x**2 + y**2

def runge_kutta_2nd_order(f, y0, x0, h, x_end):
    n_steps = int((x_end - x0) / h)
    x_vals = np.linspace(x0, x_end, n_steps + 1)
    y_vals = np.zeros(n_steps + 1)
    y_vals[0] = y0
    for i in range(n_steps):
        k1 = h * f(x_vals[i], y_vals[i])
        k2 = h * f(x_vals[i] + h / 2, y_vals[i] + k1 / 2)
        y_vals[i + 1] = y_vals[i] + k2
    return x_vals, y_vals

def runge_kutta_4th_order(f, y0, x0, h, x_end):
    n_steps = int((x_end - x0) / h)
    x_vals = np.linspace(x0, x_end, n_steps + 1)
    y_vals = np.zeros(n_steps + 1)
    y_vals[0] = y0
    for i in range(n_steps):
        k1 = h * f(x_vals[i], y_vals[i])
        k2 = h * f(x_vals[i] + h / 2, y_vals[i] + k1 / 2)
        k3 = h * f(x_vals[i] + h / 2, y_vals[i] + k2 / 2)
        k4 = h * f(x_vals[i] + h, y_vals[i] + k3)
        y_vals[i + 1] = y_vals[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x_vals, y_vals