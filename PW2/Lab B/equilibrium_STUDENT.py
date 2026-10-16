"""
PW2 Lab B Part 4 -- chemical equilibrium via the equilibrium constant K.

Reaction  H2 + I2 <=> 2 HI, starting from 1 mol H2 and 1 mol I2.
As the reaction proceeds by an extent x:  H2 = 1-x,  I2 = 1-x,  HI = 2x.
At equilibrium the composition satisfies the equilibrium constant
        K = [HI]^2 / ([H2][I2]) = (2x)^2 / ((1-x)(1-x)).
Given K, find the extent x. Solve it TWO ways and compare.
Run:  python equilibrium.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, minimize

K = 15.6
a = b = 1.0

# TODO 1: write k_imbalance(x) = (2x)^2/((a-x)(b-x)) - K.
#         It equals zero exactly at equilibrium.
def k_imbalance(x):
    return (2 * x)**2 / ((a - x) * (b - x)) - K

# TODO 2 (method 1): use scipy.optimize.newton to find the root of k_imbalance
#         (start x0=0.5). This is root-finding.
x_newton = newton(k_imbalance, x0=0.5)
print("Newton x:", x_newton)

# TODO 3 (method 2): use scipy.optimize.minimize to minimise k_imbalance(x)**2
#         (method "SLSQP", bounds [(0, 0.999)], x0=[0.5]). Print both answers
#         and confirm they agree.
def objective(x):
    return k_imbalance(x)**2

res = minimize(objective, x0=[0.5], method="SLSQP", bounds=[(0, 0.999)])
x_slsqp = res.x[0]
print("SLSQP x:", x_slsqp)

# TODO 4: report the equilibrium amounts (H2, I2, HI), and plot how the three
#         amounts change with the extent x, marking the equilibrium. Save
#         equilibrium.png.
H2_eq = a - x_slsqp
I2_eq = b - x_slsqp
HI_eq = 2 * x_slsqp

print(f"Equilibrium amounts: H2 = {H2_eq:.3f}, I2 = {I2_eq:.3f}, HI = {HI_eq:.3f}")

x_vals = np.linspace(0, 0.99, 200)
h2_vals = a - x_vals
i2_vals = b - x_vals
hi_vals = 2 * x_vals

plt.figure()
plt.plot(x_vals, h2_vals, label="H2")
plt.plot(x_vals, i2_vals, label="I2")
plt.plot(x_vals, hi_vals, label="HI")
plt.axvline(x_slsqp, color='red', linestyle='--', label=f'Equilibrium x={x_slsqp:.3f}')
plt.xlabel("Extent of reaction (x)")
plt.ylabel("Moles")
plt.legend()
plt.savefig("equilibrium.png")
plt.show()