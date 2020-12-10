"""
===============================
Title: Goodwin-Keen Model
Authors: Romi Lifshitz and Grant Forsythe
Reference: https://ms.mcmaster.ca/~grasselli/goodwin.html
===============================
"""
import numpy as np
from math import exp
from sympy.solvers import solve
from sympy import symbols

phil0 = 0.04/(1-0.04**2)
phil1 = 0.04**3/(1-0.04**2)

# Parameters
alpha = 0.025       # Technological growth rate
beta = 0.02         # Population growth rate
delta = 0.01        # Deprecation rate
kappa = lambda x: -0.0065 + exp(-5)*exp(20*x) # double check eqn is correct. Acceleration relation for the total real capital stock
phil_curve = lambda L: phil1/(1-L)**2-phil0   # Phillips curve
r = 0.03            # Real interest rate
v = 3               # Capital to output ratio


def goodwin(y: tuple, t: np.array, *args) -> np.array:
    omega, Lambda = y

    system = np.array([
        omega * (phil_curve(Lambda) - alpha),               #dL/dt
        Lambda * ((1 - omega) / v - alpha - beta - delta)   #dW/dt
    ])

    return system


def goodwin_keen(y: tuple, t: np.array, *args) -> np.array:
    Lambda, Omega, Debt = y

    system = np.array([
        Lambda * (kappa(1 - Omega - r * Debt) / v - alpha - beta - delta),                               # dL/dt
        Omega * (phil_curve(Lambda) - alpha),                                                        # dW/dt
        Debt * (r - kappa(1 - Omega - r * Debt)/ v + delta) + kappa(1 - Omega - r * Debt) - (1 - Omega)
    ])     # dR/dt

    return system

def eq_goodwin() -> list:
    """
    Finds the equilibrium point.
    """
    L, W = symbols('L W')

    system = np.array([L * ((1 - W) / v - alpha - beta - delta),
                       W * (phil_curve(L) - alpha)])

    return solve(system, L, W)

def eq_goodwin_keen() -> list:
    Lambda, Omega, Debt = symbols('L W D')

    system = np.array([
        Lambda * (kappa(1 - Omega - r * Debt) / v - alpha - beta - delta),  # dL/dt
        Omega * (phil_curve(Lambda) - alpha),  # dW/dt
        Debt * (r - ((kappa(1 - Omega - r * Debt)) / v + delta) + kappa(1 - Omega - r * Debt) - (1 - Omega))
    ])  # dR/dt

    return solve(system, Lambda, Omega, Debt)

