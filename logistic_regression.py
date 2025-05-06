import math

MAX_ITER = 2000000
FILE_PATH = "./loan2.csv"

def read_csv(file_path):
    x = []
    y = []
    z = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()                
            if line:
                data = line.split(",")
                if not data[0].isdigit() or not data[1].isdigit():
                    continue
                x.append(float(data[0]))
                y.append(float(data[1]))
                z.append(float(data[2]))
    return x, y, z

def calculate_logistic_regression_loss(w0, w1, w2, x1, x2, y):
    N = len(x1)
    sum = 0
    for i in range(N):
        sum += ( y[i] * (w1 * x1[i] + w2 * x2[i] + w0) - math.log(1 + math.exp(w1 * x1[i] + w2 * x2[i] + w0)) )
    return - (1 / N) * sum
    
def sigma(z):
    return 1 / (1 + math.exp(-z))

def gradient_descent(x1, x2, y, r, t, max_iters=MAX_ITER):
    w0 = 0
    w1 = 1
    w2 = 2

    N = len(x1)

    for iteration in range(max_iters):
        df_dw0 = 0
        df_dw1 = 0
        df_dw2 = 0

        for i in range(N):
            df_dw0 += 1 - y[i] - sigma(-(w1 * x1[i] + w2 * x2[i] + w0))
            df_dw1 += -y[i] * x1[i] + x1[i] * (1 - sigma(-(w1 * x1[i] + w2 * x2[i] + w0)))
            df_dw2 += -y[i] * x2[i] + x2[i] * (1 - sigma(-(w1 * x1[i] + w2 * x2[i] + w0)))

        df_dw0 /= N
        df_dw1 /= N
        df_dw2 /= N

        w0 = w0 - r * df_dw0
        w1 = w1 - r * df_dw1
        w2 = w2 - r * df_dw2

        loss = calculate_logistic_regression_loss(w0, w1, w2, x1, x2, y)
        if iteration % 100000 == 0:
            print(f"Iteration {iteration}: (w0={w0:.2f}, w1={w1:.2f}, w2={w2:.2f}), loss={loss:.2f}")
        if loss < t:
            break

    return w0, w1, w2


r = 0.00001
x1, x2, y = read_csv(FILE_PATH)
theshold = 0.00001
w0, w1, w2 = gradient_descent(x1, x2, y, r, theshold)
print(f"Final weights: w_0 = {w0}, w_1 = {w1}, w_2 = {w2}")