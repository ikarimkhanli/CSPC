"""
PW2 Lab B Part 2 -- three routes to a minimum.

Compare gradient descent, Newton, and SLSQP on two functions:
  2A: f(x) = (x-3)**2 + 1          (easy, one minimum at x=3)
  2B: g(x) = x**4 - 3*x**2 + x + 5 (harder, several stationary points)
Run:  python warmup.py
"""

import numpy as np
from scipy.optimize import minimize, newton

# ---------- 2A: Easy convex function ----------
def f(x):   return (x-3)**2 + 1
def df(x):  return 2*(x-3)
def d2f(x): return 2.0

print("--- Part 2A: Easy Convex Function ---")

# 1. Gradient descent by hand
x = 0.0
lr = 0.1
for _ in range(50):
    x = x - lr * df(x)
print(f"1. Gradient Descent result: x = {x:.4f}")

# 2. Newton's method on f'(x) = 0
x_newton = newton(df, x0=0.0, fprime=d2f)
print(f"2. Newton's method result: x = {x_newton:.4f}")

# 3. SLSQP method
res_slsqp = minimize(f, x0=0.0, method="SLSQP")
print(f"3. SLSQP result: x = {res_slsqp.x[0]:.4f}\n")


# ---------- 2B: Harder landscape ----------
def g(x):   return x**4 - 3*x**2 + x + 5
def dg(x):  return 4*x**3 - 6*x + 1
def d2g(x): return 12*x**2 - 6

print("--- Part 2B: Harder Landscape ---")

for x0 in [0.0, 2.0]:
    print(f"\nStarting from x0 = {x0}:")
    
    # 1. Gradient descent by hand for g(x)
    x = x0
    lr = 0.05
    for _ in range(100):
        x = x - lr * dg(x)
    print(f"  - Gradient Descent: x = {x:.4f}, g(x) = {g(x):.4f}")
    
    # 2. Newton's method for g'(x) = 0
    try:
        x_newton = newton(dg, x0=x0, fprime=d2g)
        g2 = d2g(x_newton)
        status = "Minimum (g'' > 0)" if g2 > 0 else "Maximum/Stationary (g'' < 0 or 0)"
        print(f"  - Newton's Method: x = {x_newton:.4f}, g''(x) = {g2:.2f} -> {status}")
    except RuntimeError:
        print("  - Newton's Method: Did not converge")
        
    # 3. SLSQP method for g(x)
    res_slsqp = minimize(g, x0=x0, method="SLSQP")
    print(f"  - SLSQP: x = {res_slsqp.x[0]:.4f}, g(x) = {res_slsqp.fun:.4f}")
