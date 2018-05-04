import numpy as np

a = np.random.uniform(0.0, 1.0, size=(1024, 128))
b = np.random.uniform(0.0, 1.0, size=(1024, 128))
c = a + b
print(c)