# First task solution
# IVAN CHABANOV

import numpy as np
import matplotlib.pyplot as plt

# The algorithms themselves:
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


def main():
    # Here you can define your custom functions
    # For example, choose those ones.
    
    # def f2(x, y):
    #     return x**2 + y**2
    
    # Define step size there.
    steps = [0.1, 0.05, 0.01, 0.005, 0.001]
    calculations_range = list(range(1, 2 + 1))
    calculations_amount = [int((len(calculations_range) - 1) / step) for step in steps[1:]]
    
    def f1(x, y):
        return x + np.cos(y)

    x_start = 1
    x_finish = 2
    y_start = 30
    
    rk2_solutions = [rk2(f1, 1, 2, )]
    
    # print(calculations_amount)
    print(rk2(f1, 1, 2, 30, 0.1)[1])
    print(rk2(f1, 1, 2, 30, 0.05)[1])
    print(rk2(f1, 1, 2, 30, 0.001)[1])
    print(rk2(f1, 1, 2, 30, 0.0005)[1])
    
    # Let's compute relative and absolute errors to check
    #   how significantly step size changes the algorithm precise.
    
    
    
    
    
    
    
if __name__ == "__main__":
    main()
    
    