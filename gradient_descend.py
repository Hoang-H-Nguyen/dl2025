def f(x):
    return x**2

def f_derivative(x):
    return 2 * x

def gradient_descent(f, f_derivative, x_a, L, iterations):
    x = x_a
    iterations = iterations
    L = L

    print("(time,x,f(x))")
    for i in range(iterations):
        x = x - L * f_derivative(x)
        print(f"({i},{x},{f(x)})")
    return x

gradient_descent(f, f_derivative, 10, 0.1, 10)
