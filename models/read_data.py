import numpy as np
import pandas as pd
import os

def read_data() -> pd.DataFrame:

    PATH = os.path.join(os.getcwd(), 'data')
    df = pd.read_csv(os.path.join(PATH,'cleaned_data.csv'))
    df.set_index(np.arange(1990,2009,1), inplace=True)
    return df.apply(lambda x: x/100)
