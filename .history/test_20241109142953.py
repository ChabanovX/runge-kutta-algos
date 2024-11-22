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

def calculate_relative_errors(rk_method, f, y0, x0, x_end, step_sizes, N_values):
    errors = {}
    previous_values = None
    
    for h in step_sizes:
        x_vals, y_vals = rk_method(f, y0, x0, h, x_end)
        if previous_values is not None:
            errors[h] = np.abs(y_vals[:10] - previous_values[:10])
        previous_values = y_vals
    
    return errors

def calculate_absolute_errors(errors):
    abs_errors = {}
    for h, rel_errors in errors.items():
        abs_errors[h] = np.max(rel_errors)
    return abs_errors


def plot_log_absolute_errors(abs_errors_2nd, abs_errors_4th):
    plt.figure(figsize=(10, 5))
    plt.plot(list(abs_errors_2nd.keys()), np.log2(list(abs_errors_2nd.values())), label='2nd Order')
    plt.plot(list(abs_errors_4th.keys()), np.log2(list(abs_errors_4th.values())), label='4th Order')
    plt.xlabel('Grid Step Size (h)')
    plt.ylabel('Log2 Absolute Error')
    plt.legend()
    plt.title('Log2 of Absolute Errors for 2nd and 4th Order Runge-Kutta Methods')
    plt.grid()
    plt.show()
    