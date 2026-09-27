import numpy as np

def simulate(N0, rate):
    if rate < 0:
        raise ValueError("Decay rate cannot be negative.")
    decayed = np.random.binomial(N0, rate)
    return [N0, N0 - decayed]

def simulate_loop(N0, rate):
    return simulate(N0, rate)
