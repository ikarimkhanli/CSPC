import numpy as np
import matplotlib.pyplot as plt

LAMBDA = 0.3

data = np.loadtxt("decay_observed.csv", delimiter=",", skiprows=1)
t = data[:, 0]
observed = data[:, 1]

N0 = observed[0]
analytical = N0 * np.exp(-LAMBDA * t)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

ax1.scatter(t, observed, color='blue', label='Observed', alpha=0.7)
ax1.set_title("Observed Data")
ax1.set_xlabel("Time (t)")
ax1.set_ylabel("Count (N)")
ax1.set_ylim(bottom=0)

ax2.plot(t, analytical, color='red', label='Analytical Law')
ax2.set_title("Analytical Law")
ax2.set_xlabel("Time (t)")

plt.tight_layout()
plt.savefig("figure.png")
print("figure.png was created succesfully!")