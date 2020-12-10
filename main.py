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
from scripts.model import goodwin, goodwin_keen, eq_goodwin, eq_goodwin_keen
from scripts.read_data import read_data
import os

v, Φ, k, r, alpha, beta, delta = symbols('v  Φ(λ) k  r alpha beta  delta')
dlam, domega, dd = symbols('λ^. ω^. d^.')

time = np.arange(0,300,0.01)    # for integrating
index = np.arange(0,120,20)           # for indexing purposes
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
        ax.plot(time, result[:, 2], label=r'd')
        ax.set_title(fr'$\lambda_0={initial[0]}$, $\omega_0={initial[1]}$, d=${initial[2]}$ ')
    else:
        ax.set_title(fr'$\omega_0={initial[0]}, \lambda_0={initial[1]}$')

    # ax.set_xlim(left=1, right=101)
    # ax.set_xticks(index)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Rate')
    ax.legend(loc=7)

    if file_name:
        PATH = os.path.join(os.getcwd(), 'imgs')
        my_file = f'{file_name}.svg'
        fig.savefig(os.path.join(PATH, my_file), dpi=1000)
    else:
        fig = plt.figure(tight_layout = True)
        axes = fig.add_subplot(111, projection = '3d')
        axes.plot3D(result[:,1],result[:,0], c='blue', label='plot', linewidth=0.35)
        axes.view_init(35,-25)
        # axes.scatter(0.9686, 0.8349, c='red', label=r'($\omega^*,\lambda^*$)')
        axes.set_title(fr'$\omega_0={initial[0]}, \lambda_0={initial[1]}, d_0={initial[2]}$')
        axes.set_xlabel(r'$\omega$')
        axes.set_ylabel(r'$\lambda$')
        # axes.legend(loc=1)

        PATH = os.path.join(os.getcwd(), 'imgs')
        fig.savefig(os.path.join(PATH, 'goodwin_eq.svg'), dpi=1000)
        plt.show()

## TODO: Fix this up

if __name__ == '__main__':
    plot(goodwin_keen, (0.75, 0.75,0.1))
    # Lambda_star, Omega_star = eq_goodwin()[1]
    # print(eq_goodwin_keen())
    # simulate()









