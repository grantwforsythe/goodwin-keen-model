#!/usr/bin/env python3
# TODO: equations
# TODO: solve equations
# TODO: adjust tolerance to 1e-6

import numpy as np
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(6, 4))
import scipy.integrate as integrate
import scipy.optimize as optimize
from sympy.solvers import solve
from sympy import symbols
from scripts.model import goodwin, goodwin_keen, eq_goodwin
from scripts.read_data import read_data
import os

v, Φ, k, r, alpha, beta, delta = symbols('v  Φ(λ) k  r alpha beta  delta')
dlam, domega, dd = symbols('λ^. ω^. d^.')

time = np.arange(0,800,0.01)    # for integrating
t = np.arange(0,19,1)           # for indexing purposes
df = read_data()


def plot(func: object, initial: tuple, file_name: str = '') -> None:
    """

    Simulating a macro economy using Goodwin's or a Goodwin-Keen economic model.

    :param func: goodwin or goodwin_keen.
    :param file_name: The name of the image.
    :param data: The employment rate and wage rate from the dataset.
    :param initial: The initial conditions: goodwin, (0.8,0.9), goodwin_keen (0.8,0.9,0.5).
    :return: Nothing to return.
    """

    try:
        result = integrate.odeint(func, initial, t=time)
    except:
        raise ValueError('Not a valid function or set of initial conditions.')

    fig, ax = plt.subplots(tight_layout=True)
    ax.plot(time, result[:, 0], label=r'$\omega$')
    ax.plot(time, result[:, 1], label=r'$\lambda$')

    if func == goodwin_keen:
        ax.plot(time, result[:, 2], label=r'Debt')

    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Rate')
    ax.legend(loc=1)

    if file_name:
        PATH = os.path.join(os.getcwd(), 'imgs')
        my_file = f'{file_name}.svg'
        fig.savefig(os.path.join(PATH, my_file), dpi=1000)
    else:
        plt.show()

## TODO: Fix this up

def errors(pars: tuple) -> int:
    alpha, v = pars
    sse = 0.0

    y_hats = integrate.odeint(goodwin_keen, y0=[0.75, 0.75, 0.5], t=t, args=(alpha, v))  # ??Why is shape of y_hats = (28000, 3)

    for act, est in zip(df.Lambda.values, y_hats[:,0]):
        sse += (act - est) ** 2  # ?? Do I optimize all 3 equations at once or just one of them?

    return sse


# def simulate():
#     quick_gen = (
#         optimize.minimize(errors, (p, q), method='Nelder-Mead').x
#         for p in np.linspace(0, 0.5, 5)
#         for q in np.linspace(1.5, 4.5, 5)
#     )
#
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#
#     for x, y in quick_gen:
#         print(x,y)
#         ax1.hist(x, bins=100)
#         ax2.hist(y, bins=100)
#     ax1.set_title(r'$\alpha$')
#     ax2.set_title('r')
#     plt.show()


if __name__ == '__main__':
    plot(goodwin_keen, (0.8, 0.9,0.5),'goodwin_plot')
    Lambda_star, Omega_star = eq_goodwin()[1]
    print(Lambda_star, Omega_star)
    # simulate()









