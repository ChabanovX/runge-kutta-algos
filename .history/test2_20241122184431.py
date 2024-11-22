import numpy as np
import matplotlib.pyplot as plt

# Define the ODEs
def f_a(x, y):
    return x + np.cos(y)

def f_b(x, y):
    return x**2 + y**2

# RK2 method
def rk2(f, x0, y0, h, N):
    x = np.linspace(x0, x0 + N*h, N+1)
    y = np.zeros(N+1)
    y[0] = y0
    for n in range(N):
        k1 = f(x[n], y[n])
        k2 = f(x[n] + h/2, y[n] + h/2 * k1)
        y[n+1] = y[n] + h * k2
    return x, y

# RK4 method
def rk4(f, x0, y0, h, N):
    x = np.linspace(x0, x0 + N*h, N+1)
    y = np.zeros(N+1)
    y[0] = y0
    for n in range(N):
        k1 = f(x[n], y[n])
        k2 = f(x[n] + h/2, y[n] + h/2 * k1)
        k3 = f(x[n] + h/2, y[n] + h/2 * k2)
        k4 = f(x[n] + h, y[n] + h * k3)
        y[n+1] = y[n] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return x, y

# Parameters
equations = {'a': (f_a, 1, 30), 'b': (f_b, 2, 1)}
methods = {'RK2': rk2, 'RK4': rk4}
hs = [0.1, 0.05, 0.01, 0.005, 0.001]
Ns = [int(1/h) for h in hs]  # Since interval is from x0 to x0 + 1

# Main computation
for eq_label, (f, x0, y0) in equations.items():
    print(f"\nEquation ({eq_label}):")
    solutions = {}
    for method_label, method in methods.items():
        print(f"\nMethod: {method_label}")
        absolute_errors = []
        prev_N = None
        prev_y = None
        for h, N in zip(hs, Ns):
            x, y = method(f, x0, y0, h, N)
            solutions[N] = (x, y)
            # Compute and print relative errors
            if prev_N is not None:
                # Interpolate to the coarser grid
                y_prev_interp = np.interp(x[::int(prev_N/N)], x, y)
                rel_errors = np.abs(y_prev_interp - prev_y)
                # Print errors
                print(f"N_{prev_N}:")
                print(' '.join(map(str, rel_errors)))
                # Compute absolute error
                abs_error = np.max(rel_errors)
                absolute_errors.append(abs_error)
            prev_N = N
            prev_y = y
        # Plotting absolute errors
        plt.plot(np.log2(absolute_errors), label=f"{method_label} (Eq {eq_label})")

plt.xlabel('Step Index')
plt.ylabel('log2(Absolute Error)')
plt.legend()
plt.title('Absolute Errors for RK Methods')
plt.show()
