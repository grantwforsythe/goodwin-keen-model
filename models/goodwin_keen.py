<<<<<<< Updated upstream:models/goodwin_keen.py
import os
import math
import numpy as np
import sympy as sym
from .parameters import *
import matplotlib.pyplot as plt
from sympy import symbols, Matrix
import scipy.integrate as integrate

PATH = os.path.join(os.getcwd(), 'static/imgs')

def j_keen():
    """
    The symbolic Jacobian for the Goodwin Keen Mode.
    :return:
    """
    v, r, alpha, beta, delta, phi0, phi1, kappa0, kappa1, kappa2 = symbols(
        'nu r alpha beta delta Phi_0 Phi_1 kappa_0 kappa_1 kappa_2')
    L, W, D = symbols('lambda omega d')

    phil_curve = phi1 / (1 - L) ** 2 - phi0
    kappa = kappa0 + kappa1 * sym.exp(
        kappa2 * (1 - W - r * D))  # Acceleration relation for the total real capital stock

    system = Matrix([L * (kappa / v - alpha - beta - delta),  # dL/dt
                     W * (phil_curve - alpha),  # dW/dt
                     D * (r - kappa / v + delta) + kappa - (1 - W)])  # dD/dt
    funcs = Matrix([L, W, D])

    J = system.jacobian(funcs)
    return (J)


def num_keen(y: list, t: np.array) -> list:

    W, L, D = y
    pi = 1 - W - r * D
    kappa = kappa0 + kappa1 * math.exp(kappa2 * pi)  # Acceleration relation for the total real capital stock
    phil_curve = phi1 / (1 - L) ** 2 - phi0  # Phillips curve

    system = np.array([
        W * (phil_curve - alpha),  # dW/dt
        L * (kappa / v - alpha - beta - delta),  # dL/dt
        D * (r - kappa / v + delta) + kappa - (1 - W)])  # dD/dt

    return system


def find_eqm_keen(ICs: list, t: np.array) -> list:
    """
    Finds the equilibrium points for the Goodwin Keen mode.
    :param ICs: Initial conditions
    :param t: Time
    :return: Returns the equilibrium points.
    """
    result = integrate.odeint(num_keen, ICs, t)

    W_sol = result[-50:-1, 0]  # Wage Share
    L_sol = result[-50:-1, 1]  # Employment Rate
    D_sol = result[-50:-1, 2]  # Debt Ratio

    W_eqm = sum(W_sol) / len(W_sol)
    L_eqm = sum(L_sol) / len(L_sol)
    D_eqm = sum(D_sol) / len(D_sol)

    return ([W_eqm, L_eqm, D_eqm])


def eig_keen_val(ICs: list):
    """
    Phillips Curve Parameters from Keen (1995).

    Evaluated Eigenvalues for the Goodwin Keen Model

    :param ICs: List of an initial conditions.
    :return: The eigen values of the system.
    """
    k = 0.05  # Acceleration relation for the total real capital stock
    t_eqm = np.arange(0, 1000, 0.01)
    W, L, D = find_eqm_keen(ICs, t_eqm)

    J = j_keen()

    J_eval = J.subs([('nu', v),
                     ('r', r),
                     ('alpha', alpha),
                     ('beta', beta),
                     ('delta', delta),
                     ('k', k),
                     ('Phi_1', phi1),
                     ('Phi_0', phi0),
                     ('kappa_0', kappa0),
                     ('kappa_1', kappa1),
                     ('kappa_2', kappa2),
                     ('lambda', L),
                     ('omega', W),
                     ('d', D)])
    J_eig = J_eval.eigenvects()
    eigenvals = [J_eig[0][0], J_eig[1][0], J_eig[2][0]]

    return (eigenvals)


def plot_num_keen(t: np.array, ICs: list, file_name: str = '') -> None:
    result = integrate.odeint(num_keen, ICs, t)

    W_sol = result[:, 0]  # Wage Share
    L_sol = result[:, 1]  # Employment Rate
    D_sol = result[:, 2]  # Debt Ratio

    golden_ratio = (1+np.sqrt(5))/2
    H = 4.4
    fig, ax = plt.subplots(figsize=(int(golden_ratio*H),H), tight_layout=True)
    ax.set_title(fr'$\omega_0={ICs[0]}, \lambda_0={ICs[1]}, d_0={ICs[2]}$')
    ax.set_xlim( left=t[0], right=t[-1]-1)
    
    lns_L = ax.plot(t, W_sol, label=r'$\omega$')
    lns_W = ax.plot(t, L_sol, label=r'$\lambda$')
    ax.set_ylabel('Rates')
    
    ax1 = ax.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'C2'
    ax1.set_ylabel('Ratio', color=color)  # we already handled the x-label with ax1
    lns_D = ax1.plot(t, D_sol, label=r'$d$', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax.set_xlabel('Time (years)')
    lns = lns_L + lns_W + lns_D
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=1)

    if file_name:
        my_file = f'{file_name}.svg'
        fig.savefig(os.path.join(PATH, my_file), dpi=1000)

    plt.show()


def phase_plot_keen(t: np.array, ICs: list, eqm: tuple, file_name: str = '') -> None:
    result = integrate.odeint(num_keen, ICs, t)

    W_sol = result[:, 0]  # Wage Share
    L_sol = result[:, 1]  # Employment Rate
    D_sol = result[:, 2]  # Debt Ratio

    golden_ratio = (1+np.sqrt(5))/2
    H = 4.4
    fig, axes = plt.subplots(figsize=(int(golden_ratio*H),H), tight_layout=True)
    axes.set_axis_off()
    axes = fig.add_subplot(111, projection = '3d')
    axes.plot3D(W_sol,L_sol,D_sol, c='blue', label='Transient', linewidth=0.5)
    axes.view_init(45,-20)
    axes.scatter(eqm[0], eqm[1], eqm[2], c='red', label=r'($\lambda^*, \omega^*, d^*$)')
    axes.scatter(ICs[0], ICs[1], ICs[2], c='green', label=r'($\lambda_0, \omega_0, d_0$)')
    axes.set_title(fr'$\omega_0={ICs[0]}, \lambda_0={ICs[1]}, d_0={ICs[2]}$')
    axes.set_xlabel(r'$\lambda$')
    axes.set_ylabel(r'$\omega$')
    axes.set_zlabel(r'$d$')
    axes.legend(loc=0)

    if file_name:
        my_file = f'{file_name}.svg'
        fig.savefig(os.path.join(PATH, my_file), dpi=1000)

=======
import numpy as np
import math
import scipy.integrate as integrate

import matplotlib.pyplot as plt

import sympy as sym
from sympy import symbols
from sympy import  Matrix

import os

PATH = os.path.join(os.getcwd(), 'imgs')

alpha = 0.025  # Technological growth rate
beta = 0.02  # Population growth rate
delta = 0.01  # Deprecation rate
# Phillips Curve Parameters from Keen (1995)
phi0 = 0.04 / (1 - 0.04 ** 2)
phi1 = 0.04 ** 3 / (1 - 0.04 ** 2)
# Investment Rate Parameters from Grasselli (2012)
kappa0 = -0.0065
kappa1 = math.exp(-5)
kappa2 = 20
r = 0.03  # Real interest rate
v = 3  # Capital to output ratio


def j_keen():
    """
    The symbolic Jacobian for the Goodwin Keen Mode.
    :return:
    """
    v, r, alpha, beta, delta, phi0, phi1, kappa0, kappa1, kappa2 = symbols(
        'nu r alpha beta delta Phi_0 Phi_1 kappa_0 kappa_1 kappa_2')
    L, W, D = symbols('lambda omega d')

    phil_curve = phi1 / (1 - L) ** 2 - phi0
    kappa = kappa0 + kappa1 * sym.exp(
        kappa2 * (1 - W - r * D))  # Acceleration relation for the total real capital stock

    system = Matrix([L * (kappa / v - alpha - beta - delta),  # dL/dt
                     W * (phil_curve - alpha),  # dW/dt
                     D * (r - kappa / v + delta) + kappa - (1 - W)])  # dD/dt
    funcs = Matrix([L, W, D])

    J = system.jacobian(funcs)
    return (J)


def num_keen(y: list, t: np.array) -> list:

    W, L, D = y
    pi = 1 - W - r * D
    kappa = kappa0 + kappa1 * math.exp(kappa2 * pi)  # Acceleration relation for the total real capital stock
    phil_curve = phi1 / (1 - L) ** 2 - phi0  # Phillips curve

    system = np.array([
        W * (phil_curve - alpha),  # dW/dt
        L * (kappa / v - alpha - beta - delta),  # dL/dt
        D * (r - kappa / v + delta) + kappa - (1 - W)])  # dD/dt

    return system


def find_eqm_keen(ICs: list, t: np.array) -> list:
    """
    Finds the equilibrium points for the Goodwin Keen mode.
    :param ICs: Initial conditions
    :param t: Time
    :return: Returns the equilibrium points.
    """
    result = integrate.odeint(num_keen, ICs, t)

    W_sol = result[-50:-1, 0]  # Wage Share
    L_sol = result[-50:-1, 1]  # Employment Rate
    D_sol = result[-50:-1, 2]  # Debt Ratio

    W_eqm = sum(W_sol) / len(W_sol)
    L_eqm = sum(L_sol) / len(L_sol)
    D_eqm = sum(D_sol) / len(D_sol)

    return ([W_eqm, L_eqm, D_eqm])


def eig_keen_val(ICs: list):
    """
    Phillips Curve Parameters from Keen (1995).

    Evaluated Eigenvalues for the Goodwin Keen Model

    :param ICs: List of an initial conditions.
    :return: The eigen values of the system.
    """
    k = 0.05  # Acceleration relation for the total real capital stock
    t_eqm = np.arange(0, 1000, 0.01)
    W, L, D = find_eqm_keen(ICs, t_eqm)

    J = j_keen()

    J_eval = J.subs([('nu', v),
                     ('r', r),
                     ('alpha', alpha),
                     ('beta', beta),
                     ('delta', delta),
                     ('k', k),
                     ('Phi_1', phi1),
                     ('Phi_0', phi0),
                     ('kappa_0', kappa0),
                     ('kappa_1', kappa1),
                     ('kappa_2', kappa2),
                     ('lambda', L),
                     ('omega', W),
                     ('d', D)])
    J_eig = J_eval.eigenvects()
    eigenvals = [J_eig[0][0], J_eig[1][0], J_eig[2][0]]

    return (eigenvals)


def plot_num_keen(t: np.array, ICs: list, file_name: str = '') -> None:
    result = integrate.odeint(num_keen, ICs, t)

    W_sol = result[:, 0]  # Wage Share
    L_sol = result[:, 1]  # Employment Rate
    D_sol = result[:, 2]  # Debt Ratio

    golden_ratio = (1+np.sqrt(5))/2
    H = 4.4
    fig, ax = plt.subplots(figsize=(int(golden_ratio*H),H), tight_layout=True)
    ax.set_title(fr'$\omega_0={ICs[0]}, \lambda_0={ICs[1]}, d_0={ICs[2]}$')
    ax.set_xlim( left=t[0], right=t[-1]-1)
    
    lns_L = ax.plot(t, W_sol, label=r'$\omega$')
    lns_W = ax.plot(t, L_sol, label=r'$\lambda$')
    ax.set_ylabel('Rates')
    
    ax1 = ax.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'darkgreen'
    ax1.set_ylabel('Ratio', color=color)  # we already handled the x-label with ax1
    lns_D = ax1.plot(t, D_sol, label=r'$d$', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax.set_xlabel('Time (years)')
    lns = lns_L + lns_W + lns_D
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=1)

    if file_name:
        my_file = f'{file_name}.svg'
        fig.savefig(os.path.join(PATH, my_file), dpi=1000)

    plt.show()


def phase_plot_keen(t: np.array, ICs: list, eqm: tuple, file_name: str = '') -> None:
    result = integrate.odeint(num_keen, ICs, t)

    W_sol = result[:, 0]  # Wage Share
    L_sol = result[:, 1]  # Employment Rate
    D_sol = result[:, 2]  # Debt Ratio

    golden_ratio = (1+np.sqrt(5))/2
    H = 4.4
    fig, axes = plt.subplots(figsize=(int(golden_ratio*H),H), tight_layout=True)
    axes.set_axis_off()
    axes = fig.add_subplot(111, projection = '3d')
    axes.plot3D(W_sol,L_sol,D_sol, c='blue', label='Transient', linewidth=0.5)
    # axes.view_init(45,-20)
    axes.scatter(eqm[0], eqm[1], eqm[2], c='red', label=r'($\lambda^*, \omega^*, d^*$)')
    axes.scatter(ICs[0], ICs[1], ICs[2], c='green', label=r'($\lambda_0, \omega_0, d_0$)')
    axes.set_title(fr'$\omega_0={ICs[0]}, \lambda_0={ICs[1]}, d_0={ICs[2]}$')
    axes.set_xlabel(r'$\lambda$')
    axes.set_ylabel(r'$\omega$')
    axes.set_zlabel(r'$d$')
    axes.legend(loc=0)

    if file_name:
        my_file = f'{file_name}.svg'
        fig.savefig(os.path.join(PATH, my_file), dpi=1000)

    plt.show()

def _2D_phase_plot_keen(t: np.array, ICs: list, eqm: tuple, file_name: str = '') -> None:
    result = integrate.odeint(num_keen, ICs, t)

    W_sol = result[:, 0]  # Wage Share
    L_sol = result[:, 1]  # Employment Rate
    D_sol = result[:, 2]  # Debt Ratio

    golden_ratio = (1+np.sqrt(5))/2
    H = 4.4
    fig, axes = plt.subplots(figsize=(int(golden_ratio*H),H), tight_layout=True)
    axes.set_axis_off()
    axes = fig.add_subplot()
    axes.plot(W_sol,L_sol, c='blue', label='Transient', linewidth=0.5)
    # axes.view_init(45,-20)
    # axes.scatter(eqm[0], eqm[1], eqm[2], c='red', label=r'($\lambda^*, \omega^*, d^*$)')
    axes.scatter(ICs[0], ICs[1], ICs[2], c='green', label=r'($\lambda_0, \omega_0, d_0$)')
    axes.set_title(fr'$\omega_0={ICs[0]}, \lambda_0={ICs[1]}, d_0={ICs[2]}$')
    axes.set_xlabel(r'$\lambda$')
    axes.set_ylabel(r'$\omega$')
    # axes.set_zlabel(r'$d$')
    axes.legend(loc=0)

    if file_name:
        my_file = f'{file_name}.svg'
        fig.savefig(os.path.join(PATH, my_file), dpi=1000)
>>>>>>> Stashed changes:scripts/goodwin_keen.py
    plt.show()