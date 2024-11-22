import numpy as np
import matplotlib.pyplot as plt

def runge_kutta(f, x0, y0, x_end, h, order=4):
    """
    Solves ODE y' = f(x, y) using Runge-Kutta method of specified order.

    Parameters:
    - f: function f(x, y)
    - x0: initial x value
    - y0: initial y value (scalar or array)
    - x_end: end x value
    - h: step size
    - order: order of the method (2 or 4)

    Returns:
    - x: array of x values
    - y: array of y values
    """
    N = int(np.ceil((x_end - x0) / h))
    x = x0 + np.arange(N + 1) * h
    y = np.zeros((N + 1,) + np.shape(y0))
    y[0] = y0

    for i in range(N):
        xi = x[i]
        yi = y[i]

        if order == 2:
            # Midpoint Method (RK2)
            k1 = f(xi, yi)
            k2 = f(xi + h/2, yi + h * k1 / 2)
            y[i+1] = yi + h * k2
        elif order == 4:
            # Classical Runge-Kutta Method (RK4)
            k1 = f(xi, yi)
            k2 = f(xi + h/2, yi + h * k1 / 2)
            k3 = f(xi + h/2, yi + h * k2 / 2)
            k4 = f(xi + h, yi + h * k3)
            y[i+1] = yi + h * (k1 + 2*k2 + 2*k3 + k4)/6
        else:
            raise ValueError("Unsupported order. Choose order=2 or order=4.")
        
    return x, y

def task1a():
    # Differential equation: y' = x + cos(y), y(1) = 30, 1 ≤ x ≤ 2
    def f(x, y):
        return x + np.cos(y)

    x0, y0 = 1, 30
    x_end = 2
    hs = [0.1, 0.05, 0.01, 0.005, 0.001]
    Ns = [int((x_end - x0) / h) for h in hs]

    methods = {
        'RK2': lambda f, x0, y0, x_end, h: runge_kutta(f, x0, y0, x_end, h, order=2),
        'RK4': lambda f, x0, y0, x_end, h: runge_kutta(f, x0, y0, x_end, h, order=4)
    }

    solutions = {}
    for method_name, method in methods.items():
        for h in hs:
            N = int((x_end - x0) / h)
            x, y = method(f, x0, y0, x_end, h)
            solutions[(method_name, N)] = (x, y)

    for method_name in methods:
        abs_errors = []
        Ns_pairs = list(zip(Ns[1:], Ns[:-1]))  # Pairs: (N_current, N_previous)

        for (N_current, N_prev) in Ns_pairs:
            x_current, y_current = solutions[(method_name, N_current)]
            x_prev, y_prev = solutions[(method_name, N_prev)]

            # Interpolate to align x values
            y_current_interp = np.interp(x_prev, x_current, y_current)

            diffs = np.abs(y_current_interp - y_prev)
            print(f"N_{N_current}")
            print(' '.join(map(str, diffs[:10])))

            abs_error = np.max(diffs)
            abs_errors.append(abs_error)

        plt.plot(np.log2(abs_errors), label=method_name)

    plt.xlabel('Iteration')
    plt.ylabel('log2(Max Absolute Error)')
    plt.title('Task 1a: Absolute Errors')
    plt.legend()
    plt.show()

def task1b():
    # Differential equation: y' = x^2 + y^2, y(2) = 1, 1 ≤ x ≤ 2
    def f(x, y):
        return x**2 + y**2

    x0, y0 = 2, 1
    x_end = 1  # Reverse direction
    hs = [-0.1, -0.05, -0.01, -0.005, -0.001]  # Negative steps for decreasing x
    Ns = [int((x_end - x0) / h) for h in hs]

    methods = {
        'RK2': lambda f, x0, y0, x_end, h: runge_kutta(f, x0, y0, x_end, h, order=2),
        'RK4': lambda f, x0, y0, x_end, h: runge_kutta(f, x0, y0, x_end, h, order=4)
    }

    solutions = {}
    for method_name, method in methods.items():
        for h in hs:
            N = int((x_end - x0) / h)
            x, y = method(f, x0, y0, x_end, h)
            N_positive = abs(N)
            solutions[(method_name, N_positive)] = (x, y)

    for method_name in methods:
        abs_errors = []
        Ns_positive = [abs(N) for N in Ns]
        Ns_pairs = list(zip(Ns_positive[1:], Ns_positive[:-1]))

        for (N_current, N_prev) in Ns_pairs:
            x_current, y_current = solutions[(method_name, N_current)]
            x_prev, y_prev = solutions[(method_name, N_prev)]

            y_current_interp = np.interp(x_prev, x_current, y_current)
            diffs = np.abs(y_current_interp - y_prev)
            print(f"N_{N_current}")
            print(' '.join(map(str, diffs[:10])))

            abs_error = np.max(diffs)
            abs_errors.append(abs_error)

        plt.plot(np.log2(abs_errors), label=method_name)

    plt.xlabel('Iteration')
    plt.ylabel('log2(Max Absolute Error)')
    plt.title('Task 1b: Absolute Errors')
    plt.legend()
    plt.show()

def task2():
    # Differential equation: y'' = y * sin(x), y(0) = 0, y'(0) = 1, 0 ≤ x ≤ 1
    def f(x, y):
        y1, y2 = y
        dy1 = y2
        dy2 = y1 * np.sin(x)
        return np.array([dy1, dy2])

    x0, y0 = 0, np.array([0, 1])
    x_end = 1
    hs = [0.1, 0.05, 0.01, 0.005, 0.001]
    Ns = [int((x_end - x0) / h) for h in hs]

    methods = {
        'RK2': lambda f, x0, y0, x_end, h: runge_kutta(f, x0, y0, x_end, h, order=2),
        'RK4': lambda f, x0, y0, x_end, h: runge_kutta(f, x0, y0, x_end, h, order=4)
    }

    solutions = {}
    for method_name, method in methods.items():
        for h in hs:
            N = int((x_end - x0) / h)
            x, y = method(f, x0, y0, x_end, h)
            solutions[(method_name, N)] = (x, y)

    for method_name in methods:
        abs_errors = []
        Ns_pairs = list(zip(Ns[1:], Ns[:-1]))

        for (N_current, N_prev) in Ns_pairs:
            x_current, y_current = solutions[(method_name, N_current)]
            x_prev, y_prev = solutions[(method_name, N_prev)]

            # Interpolate both components of y
            y_current_interp = np.zeros_like(y_prev)
            y_current_interp[:, 0] = np.interp(x_prev, x_current, y_current[:, 0])
            y_current_interp[:, 1] = np.interp(x_prev, x_current, y_current[:, 1])

            diffs = np.abs(y_current_interp[:, 0] - y_prev[:, 0])
            print(f"N_{N_current}")
            print(' '.join(map(str, diffs[:10])))

            abs_error = np.max(diffs)
            abs_errors.append(abs_error)

        plt.plot(np.log2(abs_errors), label=method_name)

    plt.xlabel('Iteration')
    plt.ylabel('log2(Max Absolute Error)')
    plt.title('Task 2: Absolute Errors')
    plt.legend()
    plt.show()

# Run the tasks
task1a()
task1b()
task2()