import numpy as np
import matplotlib.pyplot as plt


def rk2(f, x0, y0, x_end, h):
    N = int((x_end - x0) / h)
    x = np.linspace(x0, x_end, N+1)
    y = np.zeros((N+1,) + np.shape(y0))
    y[0] = y0
    
    for i in range(N):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h/2, y[i] + k1/2)
        y[i+1] = y[i] + k2
        
    return x, y


def rk4(f, x0, y0, x_end, h):
    N = int((x_end - x0) / h)
    x = np.linspace(x0, x_end, N+1)
    y = np.zeros((N+1,) + np.shape(y0))
    y[0] = y0
    
    for i in range(N):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h/2, y[i] + k1/2)
        k3 = h * f(x[i] + h/2, y[i] + k2/2)
        k4 = h * f(x[i] + h, y[i] + k3)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4)/6
        
    return x, y

def task1a():
    
    def f(x, y):
        return x + np.cos(y)
    
    x0, y0 = 1, 30
    x_end = 2
    hs = [0.1, 0.05, 0.01, 0.005, 0.001]
    Ns = [int((x_end - x0)/h) for h in hs]
    methods = {'RK2': rk2, 'RK4': rk4}
    solutions = {}
    for method_name, method in methods.items():
        prev_y = None
        abs_errors = []
        for h, N in zip(hs, Ns):
            x, y = method(f, x0, y0, x_end, h)
            solutions[(method_name, N)] = (x, y)
            if prev_y is not None:
                # Find common x values
                x_common = x[::int(len(x)/len(prev_x))]
                y_current = y[::int(len(y)/len(prev_y))]
                y_prev_interp = prev_y
                diffs = np.abs(y_current - y_prev_interp)
                # Print arrays
                print(f"N_{N}")
                print(' '.join(map(str, diffs[:10])))
                abs_error = np.max(diffs)
                abs_errors.append(abs_error)
            prev_x, prev_y = x, y
        # Plot log2 of absolute errors
        plt.figure()
        plt.plot(np.log2(abs_errors), label=method_name)
    plt.xlabel('Iteration')
    plt.ylabel('log2(Max Absolute Error)')
    plt.title('Task 1a: Absolute Errors')
    plt.legend()
    plt.show()

def task1b():
    def f(x, y):
        return x**2 + y**2
    x0, y0 = 2, 1
    x_end = 1
    hs = [0.1, 0.05, 0.01, 0.005, 0.001]
    Ns = [int((x0 - x_end)/h) for h in hs]
    methods = {'RK2': rk2, 'RK4': rk4}
    solutions = {}
    for method_name, method in methods.items():
        prev_y = None
        abs_errors = []
        for h, N in zip(hs, Ns):
            x, y = method(f, x0, y0, x_end, -h)
            solutions[(method_name, N)] = (x, y)
            if prev_y is not None:
                # Find common x values
                x_common = x[::int(len(x)/len(prev_x))]
                y_current = y[::int(len(y)/len(prev_y))]
                y_prev_interp = prev_y
                diffs = np.abs(y_current - y_prev_interp)
                # Print arrays
                print(f"N_{N}")
                print(' '.join(map(str, diffs[:10])))
                abs_error = np.max(diffs)
                abs_errors.append(abs_error)
            prev_x, prev_y = x, y
        # Plot log2 of absolute errors
        plt.figure()
        plt.plot(np.log2(abs_errors), label=method_name)
    plt.xlabel('Iteration')
    plt.ylabel('log2(Max Absolute Error)')
    plt.title('Task 1b: Absolute Errors')
    plt.legend()
    plt.show()

def task2():
    def f(x, y):
        y1, y2 = y
        dy1 = y2
        dy2 = y1 * np.sin(x)
        return np.array([dy1, dy2])
    
    x0, y0 = 0, np.array([0, 1])
    x_end = 1
    hs = [0.1, 0.05, 0.01, 0.005, 0.001]
    Ns = [int((x_end - x0)/h) for h in hs]
    methods = {'RK2': rk2, 'RK4': rk4}
    solutions = {}
    
    for method_name, method in methods.items():
        prev_y = None
        abs_errors = []
        for h, N in zip(hs, Ns):
            x, y = method(f, x0, y0, x_end, h)
            solutions[(method_name, N)] = (x, y)
            if prev_y is not None:
                # Find common x values
                x_common = x[::int(len(x)/len(prev_x))]
                y_current = y[::int(len(y)/len(prev_y))]
                y_prev_interp = prev_y
                diffs = np.abs(y_current[:,0] - y_prev_interp[:,0])
                # Print arrays
                print(f"N_{N}")
                print(' '.join(map(str, diffs[:10])))
                abs_error = np.max(diffs)
                abs_errors.append(abs_error)
            prev_x, prev_y = x, y
        # Plot log2 of absolute errors
        plt.figure()
        plt.plot(np.log2(abs_errors), label=method_name)
    plt.xlabel('Iteration')
    plt.ylabel('log2(Max Absolute Error)')
    plt.title('Task 2: Absolute Errors')
    plt.legend()
    plt.show()

# Run the tasks
# task1a()
# task1b()
task2()
