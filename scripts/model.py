"""
===============================
Title: Goodwin-Keen Model
Authors: Romi Lifshitz and Grant Forsyth
Reference: https://ms.mcmaster.ca/~grasselli/goodwin.html
===============================
"""
import numpy as np

phil0 = 0.04/(1-0.04**2)
phil1 = 0.04**3/(1-0.04**2)

# Constants
alpha = 0.025       # Technological growth rate
beta = 0.02         # Population growth rate
delta = 0.01        # Deprecation rate
k = 0.05            # Acceleration relation for the total real capital stock
phil_curve = lambda x: phil1/(1-x)**2-phil0   # Phillips curve
r = 0.03            # Real interest rate
v = 3               # Capital to output ratio


def goodwin(y: list, t: np.array) -> np.array:
    omega, Lambda = y

    system = np.array([
        omega * (phil_curve(Lambda) - alpha),               #dL/dt
        Lambda * ((1 - omega) / v - alpha - beta - delta)   #dW/dt
    ])

    return system


def goodwin_keen(y: list, t: np.array) -> np.array:
    L, W, D = y

    system = np.array([
        L * (k * (1 - W - r * D) / v - alpha - beta - delta),                               # dL/dt
        W * (phil_curve(L) - alpha),                                                        # dW/dt
        D * (r - ((k * (1 - W - r * D)) / v + delta) + k * (1 - W - r * D) - (1 - W))])     # dR/dt

    return (system)

