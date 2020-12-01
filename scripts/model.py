"""
===============================
Title: Goodwin-Keen Model
Authors: Romi Lifshitz and Grant Forsythe
===============================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import os
years = np.arange(1990,2009,1)

def read_data():
    PATH = os.path.join(os.getcwd(), 'data')
    return pd.read_csv(os.path.join(PATH,'cleaned_data.csv'))

percent = lambda x: x/100
df  = read_data()
df.set_index(years,inplace=True)
print(df.apply(percent))

# Constants
alpha = 0.025       # Technological growth rate
beta = 0.02         # Population growth rate
delta = 0.01        # Deprecation rate
k = None            # Acceleration relation for the total real capital stock
phil_curve = None   # Phillips curve
r = 0.03            # Real interest rate
v = 3               # Capital to output ratio
