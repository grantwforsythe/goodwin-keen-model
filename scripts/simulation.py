import numpy as np

import matplotlib.pyplot as plt

import sympy as sym

from scripts.goodwin_keen import find_eqm_keen, eig_keen_val


def sim_study():
    L_IC = np.linspace(0.6, 1, 10)
    W_IC = np.linspace(0.5, 1, 10)
    D_IC = np.linspace(0, 10, 10)

    results = []

    for i in range(len(L_IC)):
        for j in range(len(W_IC)):
            for k in range(len(D_IC)):
                ICs = [L_IC[i], W_IC[j], D_IC[k]]
                # DETERMINE EQUILIBRIUM
                t_eqm = np.arange(0, 1000, 0.01)
                eqm = find_eqm_keen(ICs, t_eqm)
                EQms = [eqm[1], eqm[0], eqm[2]]
                try:
                    Eig = eig_keen_val(ICs)
                    eq_type = all(sym.functions.elementary.complexes.re(v) < 0 for v in Eig)
                    results.append([ICs, EQms, eq_type])  # The one at the ends indicates Good Eqm
                except:
                    results.append([ICs, EQms, False])  # The one at the ends indicates Bad Eqm

    return (results)


def sim_study_plot(sims):

    golden_ratio = (1+np.sqrt(5))/2
    H = 4.4
    fig = plt.figure(figsize=(int(golden_ratio*H),H), tight_layout = True)
    axes = fig.add_subplot(111, projection='3d')
    axes.view_init(25, -75)

    for i in range(len(sims)):
        coors = sims[i][0]
        if sims[i][2]:
            pG = axes.scatter(coors[0], coors[1], coors[2], c = 'Blue', s = 25, label = r'C')
        else:
            pB = axes.scatter(coors[0], coors[1], coors[2], c = 'Red', s = 2, label = r'D', alpha = 0.5)
    
    axes.set_xlabel(r'$\lambda$')
    axes.set_ylabel(r'$\omega$')
    axes.set_zlabel(r'$d$')
    axes.legend([pG, pB], ['Convergent', 'Divergent'], numpoints = 1)

    plt.show()