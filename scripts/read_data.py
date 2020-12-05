import pandas as pd
import os

def read_data() -> pd.DataFrame:
    """
    Reads in data stored as a csv file and returns a DataFrame
    """
    PATH = os.path.join(os.getcwd(), 'data')
    return pd.read_csv(os.path.join(PATH,'cleaned_data.csv'))
