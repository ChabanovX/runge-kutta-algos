## Numerical Solutions of Ordinary Differential Equations Using Runge-Kutta Methods

### Overview

This project provides Python implementations to numerically solve two ordinary differential equations (ODEs) using explicit Runge-Kutta (RK) methods of the 2nd and 4th orders. The solutions are computed for various grid steps, and the relative and absolute errors are analyzed to assess the accuracy of the numerical methods. Additionally, the results are visualized through plots of the logarithm base 2 of the absolute errors.

### Tasks

1. **Task 1: First-Order ODEs**
    - **Equation (a):**
      \[
      y' = x + \cos y, \quad y(1) = 30, \quad 1 \leq x \leq 2
      \]
    - **Equation (b):**
      \[
      y' = x^2 + y^2, \quad y(2) = 1, \quad 1 \leq x \leq 2
      \]

2. **Task 2: Second-Order ODE**
    - **Equation:**
      \[
      y'' = y \sin x, \quad y(0) = 0, \quad y'(0) = 1, \quad 0 \leq x \leq 1
      \]

### Features

- **Runge-Kutta Methods:** Implements both 2nd and 4th order explicit Runge-Kutta methods for solving ODEs.
- **Variable Grid Steps:** Supports grid steps \( h = 0.1, 0.05, 0.01, 0.005, 0.001 \) corresponding to \( N = 10, 20, 100, 200, 1000 \) steps.
- **Error Analysis:** Calculates relative errors between successive \( N \) values and determines absolute errors.
- **Output Formatting:** Prints arrays of relative errors in a structured format.
- **Visualization:** Generates plots of \(\log_2\) of absolute errors for both RK methods in a single figure for comparison.

### Requirements

- **Python 3.x**
- **Libraries:**
  - `numpy`
  - `matplotlib`

### Installation

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/runge-kutta-ode-solver.git
    ```
2. **Navigate to the Project Directory:**
    ```bash
    cd runge-kutta-ode-solver
    ```
3. **Install Required Libraries:**
    ```bash
    pip install numpy matplotlib
    ```
    
### Usage

1. **Run the Python Script:**
    ```bash
    python runge_kutta_solver.py
    ```
2. **Output:**
    - **Console Output:** Displays arrays of relative errors for each numerical method and task.
    - **Plots:** Shows \(\log_2\) of absolute errors for RK2 and RK4 methods in separate figures for Task 1 and Task 2.

### Example Output

**Console:**
```
Equation (a):

Method: RK2

N_10:
0.00123 0.00234
N_20:
0.00056 0.00112
...
```

**Plots:**
- **Task 1:** Plot comparing \(\log_2\) absolute errors for RK2 and RK4 methods.
- **Task 2:** Separate plot comparing \(\log_2\) absolute errors for RK2 and RK4 methods.

### Notes

- **Customization:** You can modify the grid steps, ODEs, and initial conditions directly in the `runge_kutta_solver.py` script as needed.
- **Performance:** For very large \( N \), execution time may increase. Ensure adequate computational resources are available.
- **Extensibility:** The code is modular, allowing for easy addition of higher-order Runge-Kutta methods or additional ODEs.
