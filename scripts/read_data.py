import os
import pandas as pd 

def read_data():
    PATH = os.path.join(os.getcwd(), 'data')
    
    for file in os.listdir(PATH):
        yield pd.read_csv(os.path.join(PATH,file))
