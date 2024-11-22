import numpy as np
import matplotlib.pyplot as plt


def rk2(f, x0, y0, x_end, h):
    x_values = np.arange(x0, x_end + h, h)
    y_values = np.zeros(len(x_values)) 
    y_values[0] = y0

    for i in range(1, len(x_values)):
        x_n = x_values[i - 1]
        y_n = y_values[i - 1]

        k1 = f(x_n, y_n)
        k2 = f(x_n + h, y_n + h * k1)
        y_values[i] = y_n + (h / 2) * (k1 + k2)
        

    return x_values, y_values

def rk4(f, x0, y0, x_end, h):
    x_values = np.arange(x0, x_end + h, h)
    y_values = np.zeros(len(x_values))
    y_values[0] = y0

    for i in range(1, len(x_values)):
        x_n = x_values[i - 1]
        y_n = y_values[i - 1]

        k1 = f(x_n, y_n)
        k2 = f(x_n + h / 2, y_n + h / 2 * k1)
        k3 = f(x_n + h / 2, y_n + h / 2 * k2)
        k4 = f(x_n + h, y_n + h * k3)

        y_values[i] = y_n + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_values, y_values

# Define the ODE y' = -2y + 2x^2 + 2x
def f(x, y):
    return -2 * y + 2 * x**2 + 2 * x

# Analytical solution
def exact_solution(x):
    return np.exp(-2 * x) + x**2

# Initial conditions
x0 = 0
y0 = 1
x_end = 1

# Step sizes
h_values = [0.1, 0.05, 0.01, 0.005, 0.001]

# Prepare plots
plt.figure(figsize=(14, 7))

for h in h_values:
    # RK2 method
    x_rk2, y_rk2 = rk2(f, x0, y0, x_end, h)
    y_exact_rk2 = exact_solution(x_rk2)
    error_rk2 = np.abs(y_rk2 - y_exact_rk2)

    # RK4 method
    x_rk4, y_rk4 = rk4(f, x0, y0, x_end, h)
    y_exact_rk4 = exact_solution(x_rk4)
    error_rk4 = np.abs(y_rk4 - y_exact_rk4)

    # Plot errors
    plt.plot(x_rk2, error_rk2, label=f'RK2 Error h={h}')
    # plt.plot(x_rk4, error_rk4, '--', label=f'RK4 Error h={h}')

plt.xlabel('x')
plt.ylabel('Absolute Error')
plt.title('Error Comparison of RK2 and RK4 Methods')
plt.legend()
plt.yscale('log')  # Use logarithmic scale for error
plt.grid(True)
plt.show()