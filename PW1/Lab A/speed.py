import time
import numpy as np
from decay import simulate, simulate_loop

N0 = 100_000
rate = 0.4

start = time.perf_counter()
simulate(N0, rate)
vectorized_time = time.perf_counter() - start

start = time.perf_counter()
simulate_loop(N0, rate)
loop_time = time.perf_counter() - start

print(f"Vectorized simulation time: {vectorized_time:.6f} seconds")
print(f"Loop simulation time:       {loop_time:.6f} seconds")
