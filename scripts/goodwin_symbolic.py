import numpy as np

from sympy.solvers import solve
from sympy import symbols
from sympy import  Matrix
from sympy import  simplify
from sympy import init_printing

def j_goodwin():
    """

    Symbolic Jacobian for the Goodwin model.

    :return:
    """
    v, r, alpha, beta, delta, phil0, phil1 = symbols('nu r alpha beta delta Phi_0 Phi_1')
    L, W = symbols('lambda omega')

    phil_curve = phil1 / (1 - L) ** 2 - phil0

    system = Matrix([L * ((1 - W) / v - alpha - beta - delta), W * (phil_curve - alpha)])
    funcs = Matrix([L, W])

    J = system.jacobian(funcs)
    return (J)


def eqm_goodwin():
    """

    Symbolic Equilibrium Points for the Goodwin Model.

    :return:
    """
    v, r, alpha, beta, delta, k, phi1, phi0 = symbols('nu r alpha beta delta k Phi_1 Phi_0')
    L, W = symbols('lambda omega')
    phil_curve = phi1 / (1 - L) ** 2 - phi0

    system = np.array([
        L * ((1 - W) / v - alpha - beta - delta),
        W * (phil_curve - alpha)
    ])

    return (Matrix(solve(system, L, W)))


def eig_goodwin():
    """

    Symbolic Eigenvalues and Eigenvectors for the Goodwin Model

    :return:
    """
    init_printing(forecolor='White')

    eqm = eqm_goodwin()
    J = j_goodwin()

    # eqm = [ eqm[0], eqm[1] ] # Trivial Eqm. Point
    eqm = [eqm[2], eqm[3]]  # Stable, Econmically Realistic Eqm. Point
    # eqm = [ eqm[4], eqm[5] ] # Stable, Econmically Unrealistic Eqm. Point

    J_eval = J.subs([('lambda', eqm[0]), ('omega', eqm[1])])
    J_eig = J_eval.eigenvects()
    J_eigvals = [simplify(J_eig[0][0]), simplify(J_eig[1][0])]

    return (J_eigvals)


def eig_goodwin_val():
    """
    Evaluated Eigenvalues for the Goodwin model at equilibrium.

    :return:
    """
    v, r, alpha, beta, delta, k, phi1, phi0 = symbols('nu r alpha beta delta k Phi_1 Phi_0')
    L, W = symbols('lambda omega')

    # Phillips Curve Parameters from Keen (1995)
    phi0 = 0.04 / (1 - 0.04 ** 2)
    phi1 = 0.04 ** 3 / (1 - 0.04 ** 2)
    alpha = 0.025  # Technological growth rate
    beta = 0.02  # Population growth rate
    delta = 0.01  # Deprecation rate
    k = 0.05  # Acceleration relation for the total real capital stock
    r = 0.03  # Real interest rate
    v = 3  # Capital to output ratio

    eig = eig_goodwin()

    eig_vals1 = eig[0].subs([('nu', v),
                             ('r', r),
                             ('alpha', alpha),
                             ('beta', beta),
                             ('delta', delta),
                             ('k', k),
                             ('Phi_1', phi1),
                             ('Phi_0', phi0)])

    eig_vals2 = eig[1].subs([('nu', v),
                             ('r', r),
                             ('alpha', alpha),
                             ('beta', beta),
                             ('delta', delta),
                             ('k', k),
                             ('Phi_1', phi1),
                             ('Phi_0', phi0)])

    return ([eig_vals1, eig_vals2])