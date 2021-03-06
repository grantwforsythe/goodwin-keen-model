import os
import numpy as np
from sympy.solvers import solve
import matplotlib.pyplot as plt
from sympy import symbols, Matrix
import scipy.integrate as integrate

PATH = os.path.join(os.getcwd(), 'static/imgs')

def num_goodwin(y, t):
    W, L = y

    alpha = 0.025  # Technological growth rate
    beta = 0.02  # Population growth rate
    delta = 0.01  # Deprecation rate
    phil0 = 0.04 / (1 - 0.04 ** 2)
    phil1 = 0.04 ** 3 / (1 - 0.04 ** 2)
    phil_curve = lambda x: phil1 / (1 - x) ** 2 - phil0  # Phillips curve
    v = 3  # Capital to output ratio

    system = np.array([W * (phil_curve(L) - alpha),
                       L * ((1 - W) / v - alpha - beta - delta)])

    return system


def eqm_goodwin_eval():  # Evaluated Symbolic Eqm. Points
    v, r, alpha, beta, delta, k, phi1, phi0 = symbols('v r alpha beta delta k Phi_1 Phi_0')
    L, W = symbols('lambda omega')
    phil_curve = phi1 / (1 - L) ** 2 - phi0

    system = np.array([L * ((1 - W) / v - alpha - beta - delta),
                       W * (phil_curve - alpha)])

    # Phillips Curve Parameters from Keen (1995)
    phi0 = 0.04 / (1 - 0.04 ** 2)
    phi1 = 0.04 ** 3 / (1 - 0.04 ** 2)
    alpha = 0.025  # Technological growth rate
    beta = 0.02  # Population growth rate
    delta = 0.01  # Deprecation rate
    k = 0.05  # Acceleration relation for the total real capital stock
    r = 0.03  # Real interest rate
    v = 3  # Capital to output ratio

    eqm_evaluated = Matrix(solve(system, L, W)).subs([('v', v),
                                                      ('r', r),
                                                      ('alpha', alpha),
                                                      ('beta', beta),
                                                      ('delta', delta),
                                                      ('k', k),
                                                      ('Phi_1', phi1),
                                                      ('Phi_0', phi0)])

    return (eqm_evaluated)


def plot_num_goodwin(t: tuple , ICs: tuple, file_name: str = '') -> None:
    result = integrate.odeint(num_goodwin, ICs, t)

    L_sol = result[:, 1]  # Employment Rate
    W_sol = result[:, 0]  # Wage Share

    golden_ratio = (1+np.sqrt(5))/2
    H = 4.4
    fig, ax = plt.subplots(figsize=(int(golden_ratio*H),H), tight_layout=True)
    
    ax.set_title(fr'$\omega_0={ICs[0]}, \lambda_0={ICs[1]}$')
    ax.set_xlim( left=t[0], right=t[-1]-1)
    
    color = 'C0'
    ax.set_ylabel(r'$\omega$', color=color)
    ax.plot(t, W_sol, label=r'$\omega$', color=color)
    ax.tick_params(axis='y', labelcolor=color)
    
    ax1 = ax.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'C1'
    ax1.set_ylabel(r'$\lambda$', color=color)  # we already handled the x-label with ax1
    ax.plot(t, L_sol, label=r'$\lambda$', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax.set_xlabel('Time (years)')

    if file_name:
        my_file = f'{file_name}.svg'
        fig.savefig(os.path.join(PATH, my_file), dpi=1000)

    plt.show()


def phase_plot_goodwin(t: tuple, ICs: tuple, eqm: tuple, file_name: str = '') -> None:
    """
    Phase plot for the Goodwin model
    :param t: Time
    :param ICs: Initial conditions.
    :param eqm: Equilibrium points.
    :param file_name: Name of the svg file.
    :param save: Do you want to save the file?
    :return: Nothing to return.
    """
    result = integrate.odeint(num_goodwin, ICs, t)

    L_sol = result[:, 1]  # Employment Rate
    W_sol = result[:, 0]  # Wage Share

    golden_ratio = (1+np.sqrt(5))/2
    H = 4.4
    fig, axes = plt.subplots(figsize=(int(golden_ratio*H),H), tight_layout=True)
    
    axes.plot(L_sol, W_sol, c='blue', label='Transient', linewidth=0.35)
    axes.scatter(eqm[0], eqm[1], c='red', label=r'($\lambda^*, \omega^*$)')
    
    axes.set_title(fr'$\omega_0={ICs[0]}, \lambda_0={ICs[1]}$')
    axes.set_xlabel(r'$\lambda$')
    axes.set_ylabel(r'$\omega$')
    axes.legend(loc=0)

    if file_name:
        my_file = f'{file_name}.svg'
        fig.savefig(os.path.join(PATH, my_file), dpi=1000)

    plt.show()