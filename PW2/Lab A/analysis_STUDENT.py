"""
PW2 Lab A -- Motion from tracking data.

Read noisy free-fall position measurements, then:
  - differentiate once  -> velocity
  - differentiate twice -> acceleration (should be ~ constant -g, but noisy!)
  - integrate the acceleration back up -> recover velocity and position

Complete the TODOs. Run:  python analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid


# TODO 1: read freefall.csv into arrays t and y
#         (hint: np.loadtxt with a comma delimiter, skipping the header)
data=np.loadtxt('freefall.csv',delimiter=',',skiprows=1)
t=data[:,0]
y=data[:,1]


# TODO 2: compute velocity v = derivative of y w.r.t. t   (np.gradient)
#         and acceleration a = derivative of v w.r.t. t    (np.gradient again)
#         Print the mean acceleration. Is it close to -9.81? Is it noisy?
v=np.gradient(y,t)
a=np.gradient(v,t)
mean_accel=np.mean(a)
print(f"mean acc : {mean_accel}ms/s^2")

# TODO 3: integrate a back up to recover velocity and position
#         (hint: cumulative_trapezoid(a, t, initial=0) + v[0], then again)
#diference between original and new one
v_recovered = cumulative_trapezoid(a, t, initial=0) + v[0]
y_recovered = cumulative_trapezoid(v_recovered, t, initial=0) + y[0]
max_diff = np.max(np.abs(y - y_recovered))
print(f"fixed one: {max_diff}")
# Part 3: standard deviation of accl 
acc_std = a.std()
print(f"standard deviation(Std Dev): {acc_std}")

# TODO 4: make a figure with 3 stacked panels: position, velocity, acceleration
#         vs time. Mark the true -9.81 line on the acceleration panel.
#         Save it as motion.png


# Part 5: Make a figure with 3 stacked panels

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# 1. Position panel
ax1.plot(t, y, label='Original Position', color='blue')
ax1.plot(t, y_recovered, label='Recovered Position', color='orange', linestyle='--')
ax1.set_ylabel('Position (m)')
ax1.legend()
ax1.grid(True)

# 2. Velocity panel
ax2.plot(t, v, label='Velocity', color='green')
ax2.set_ylabel('Velocity (m/s)')
ax2.legend()
ax2.grid(True)

# 3. Acceleration panel
ax3.plot(t, a, label='Acceleration', color='red', alpha=0.6)
ax3.axhline(-9.81, color='black', linestyle='--', label='True g (-9.81)')
ax3.set_ylabel('Acceleration (m/s^2)')
ax3.set_xlabel('Time (s)')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.savefig('motion.png')
print("motion.png uğurla yaradıldı və yadda saxlanıldı!")
