import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

train = pd.read_csv('train-balanced-sarcasm.csv')

train['Year'] = train['date'].str[:4]

tab = pd.crosstab(train.author, columns=train.label, margins=True)
print(tab.div(tab['All'], axis=0) * 100)
