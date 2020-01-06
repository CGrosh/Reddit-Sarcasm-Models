import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
import os 
import pickle
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Sarcasm_Models:
    def __init__(self, modelPath):
        self.model = keras.models.load_model(modelPath)

    def predict(self, txt):
        tokenfile = open('TokenFile', 'rb')
        tokenizer = pickle.load(tokenfile)
        # tokenfile.close()

        test_text = tokenizer.texts_to_sequences([txt])
        test_text = pad_sequences(test_text, maxlen=50)

        pred = self.model.predict(test_text)
        
        x = []

        if pred >= 0.5:
            x.append('Sarcastic')
        else:
            x.append('Not Sarcastic')           
        return (x, pred)

mod = Sarcasm_Models('new_saved_model.h5')
print(mod.predict(sys.argv[1]))


        