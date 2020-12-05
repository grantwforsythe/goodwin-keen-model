#!/usr/bin/env python3
# TODO: equations
# TODO: solve equations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.optimize as optimize
from scripts.model import goodwin, goodwin_keen
from scripts.read_data import read_data
import math
import os

time = np.arange(0,100,0.01)

def goodwin_plot(save: bool = False, initial: list = [0.8,0.9]) -> None:

    result = integrate.odeint(goodwin, initial, t=time)

    fig, ax = plt.subplots(figsize=(6,4), tight_layout=True)
    ax.plot(time, result[:, 0], label=r'$\lambda$')
    ax.plot(time, result[:, 1], label=r'$\omega$')
    ax.set_xlabel('Years')
    ax.set_ylabel('Rate')
    ax.legend(loc=1)

    PATH = os.path.join(os.getcwd(), 'imgs')
    my_file = 'goodwin.svg'

    if save:
        fig.savefig(os.path.join(PATH,my_file),dpi=500)
    else:
        plt.show()

def goodwin_keen_plot():
    pass

if __name__ == '__main__':

    goodwin_plot()

