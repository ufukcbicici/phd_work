import numpy as np


def a(n_, p_):
    return 1.0 / (n_ * (np.log(n_)**p_))

sequence = []
p = -3.0
for n in range(2, 1000):
    sequence.append(a(n, p))



print("X")