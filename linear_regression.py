def read_csv(file_path):
    x = []
    y = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line:
                data = line.split(",")
                x.append(float(data[0]))
                y.append(float(data[1]))
    return x, y

def calculate_loss(w0, w1, x, y):
    N = len(x)
    sum = 0
    for i in range(N):
        sum += ( (w1 * x[i] + w0) - y[i] ) ** 2
    return (1 / 2) * (1 / N) * sum

def gradient_descent(x, y, r, t, max_iters=10000):
    N = len(x)
    w0 = 0
    w1 = 1

    for iteration in range(max_iters):
        dw0 = 0
        dw1 = 0
        
        for i in range(N):
            error = (w1 * x[i] + w0) - y[i]
            dw0 += error
            dw1 += x[i] * error 

        dw0 /= N
        dw1 /= N

        w0 = w0 - (r * dw0)
        w1 = w1 - (r * dw1)
        
        loss = calculate_loss(w0, w1, x, y)
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: (w0={w0:.2f}, w1={w1:.2f}), loss={loss:.2f}")
        if loss < t:
            break
    return w0, w1        

r = 0.01
x, y = read_csv("./lr.csv")
theshold = 0.01
w0, w1 = gradient_descent(x, y, r, 0.01)
print(f"Final weights: w_0 = {w0}, w_1 = {w1}")
print(x, y)