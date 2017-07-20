import numpy as np
from numpy.linalg import inv

lr = 0.001
y = np.array(
    [126,  34,  78, 120,  83,  62, 104,   6,  70, 142, 147,  63,  35, 126,   9,  84,   7, 122,  93,  29,  95, 141,
     42, 102,  38,  96, 130,  83, 138, 148]).astype(np.float64)
x = np.arange(1, len(y)+1).astype(np.float64)
f = np.zeros(shape=(len(y),)).astype(np.float64)


def eval_polynom(f, x):
    res = 0
    for i in range(len(f)):
        res += f[i] * x ** (len(f) - i - 1)
    return res


def mse():
    errors = []
    for i in range(len(y)):
        x_val = x[i]
        y_val = y[i]
        x_est = eval_polynom(f, x_val)
        errors.append((y_val - x_est)**2.0)
    mean_err = np.array(errors).mean()
    return mean_err


def grad():
    g = np.zeros(shape=(len(y),)).astype(np.float64)

size = 30
A = np.zeros(shape=(size, size), dtype=np.float64)
for i in range(size):
    for j in range(size):
        A[i, j] = x[i] ** float(j)

B = inv(A)
p = np.dot(y, B)

ms_err = mse()
print("ms_err={0}".format(ms_err))
