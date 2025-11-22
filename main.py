import numpy as np
import time

N = 2**14

a = np.random.rand(*(N, N))
b = np.random.rand(*(N, N))

t0 = time.perf_counter_ns()
c = a @ b
t1 = time.perf_counter_ns()

t = t1 - t0

print(f'Finish in: {t} ns')
print(f'Finish in: {t / (1e9)} s')
