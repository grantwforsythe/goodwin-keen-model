import os
import pandas as pd 

def read_data():
    PATH = os.path.join(os.getcwd(), 'data')
    return pd.read_csv(os.path.join(PATH,'cleandata.csv'))
    # for file in os.listdir(PATH):
    #     yield pd.read_csv(os.path.join(PATH,file))
