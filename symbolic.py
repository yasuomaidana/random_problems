import sympy as sp

def newton_raphson(func_str, x0, tolerance=1e-6, max_iterations=100):
    """
    Finds a solution to a function using the Newton-Raphson method.

    Args:
      func_str: The function as a string, e.g., "x**2 - 2".
      x0: The initial guess for the root.
      tolerance: The desired accuracy.
      max_iterations: The maximum number of iterations.

    Returns:
      The approximate root of the function.
    """

    x = sp.symbols('x')
    func = sp.sympify(func_str)  # Convert string to symbolic expression

    # Calculate derivative using the 4-step method
    def derivative(f, x, h=1e-6):
        return (f.subs(x, x + h) - f.subs(x, x - h)) / (2 * h)

    df = derivative(func, x)  # Symbolic derivative

    for i in range(max_iterations):
        x1 = x0 - func.subs(x, x0) / df.subs(x, x0)
        if abs(x1 - x0) < tolerance:
            return x1
        x0 = x1

    return None  # Solution not found within max_iterations

if __name__ == "__main__":
    # Example usage
    function_str = "x**3 - 2*x - 5"
    initial_guess = 5.0

    solution = newton_raphson(function_str, initial_guess)

    if solution is not None:
        print(f"Approximate solution: {solution}")
    else:
        print("Solution not found.")