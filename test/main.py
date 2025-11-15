N = 2 ** 14

import numpy as np
import time

a = np.random.rand(N, N).astype(np.float32)
b = np.random.rand(N, N).astype(np.float32)

start = time.perf_counter_ns()
c = a@b
end = time.perf_counter_ns()
print("Time taken:", end - start)