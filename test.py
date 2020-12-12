import numpy as np

# 100 Years Goodwin Model (Numerical Solution)
from scripts.goodwin import plot_num_goodwin, eqm_goodwin_eval, phase_plot_goodwin
from scripts.goodwin_keen import find_eqm_keen, phase_plot_keen, plot_num_keen
from scripts.goodwin_symbolic import j_goodwin, eqm_goodwin, eig_goodwin, eig_goodwin_val
from scripts.simulation import sim_study, sim_study_plot

if __name__ == '__main__':
    J_g = j_goodwin()
    Eqm_g = eqm_goodwin()
    Eigvals_g = eig_goodwin()
    Eig_eval_g = eig_goodwin_val()

    print(f'Symbolic Jacobian Matrix for the Goodwin Mode:\n{J_g}\n')
    print(f'Symbolic Eqm. Points for the Goodwin Model:\n{Eqm_g}\n')
    print(f'Symbolic Eigenvalues for the Goodwin Model:\n{Eigvals_g}\n')
    print(f'Eigenvalues for the Goodwin Model Evaluated at Eqm.:\n{Eig_eval_g}\n')

    t = np.arange(0,120,0.01) # 100 Year Model
    ICs = [0.8,0.9]
    plot_num_goodwin(t, ICs, 'goodwin_model')

    # Long-Term Behaviour
    t = np.arange(0,1002,0.01) # 800 Year Model
    ICs = [0.8,0.9]
    plot_num_goodwin(t, ICs, 'goodwin_longterm')

    # Goodwin Phase Portrait
    eqm_values = eqm_goodwin_eval()
    # eqm = [ eqm_values[0], eqm_values[1] ] # Trivial Eqm. Point
    eqm = [ eqm_values[2], eqm_values[3] ] # Stable, Econmically Realistic Eqm. Point
    # eqm = [ eqm_values[4], eqm_values[5] ] # Stable, Econmically Unrealistic Eqm. Point
    t = np.arange(0,1000,0.01) # 100 Year Model
    ICs = [0.8,0.9]
    phase_plot_goodwin(t, ICs, eqm)

    t = np.arange(0, 302, 0.01)  # 100 Year Model
    ICs = [0.75, 0.75, 0.1]
    plot_num_keen(t, ICs, 'keen_model')

    # # Goodwin-Keen Phase Portrait
    t = np.arange(0, 302, 0.01)
    t_eqm = np.arange(0, 1000, 0.01)
    eqm = find_eqm_keen(ICs, t_eqm)

    phase_plot_keen(t, ICs, eqm)
    print(eqm)

    sims = sim_study()

    converge = 0
    diverge = 0
    total = len(sims)
    for i in range(total):
        if sims[i][2]:
            converge += 1
        else:
            diverge += 1

    print('Total Convergence Events: ', converge)
    print('Total Divergence Events: ', diverge)
    print('Total Events: ', total)

    sim_study_plot(sims)