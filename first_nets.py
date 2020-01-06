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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = pd.read_csv('train-balanced-sarcasm.csv')
data['comment'] = data['comment'].fillna('word')
cols = list(data)
cols.remove('label')

data_x = data[cols]
data_y = data['label']

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, 
                                    test_size=0.3, random_state=4)

maxlen = 50
training_samples = len(x_train)
max_words = 10000

# x_train['comment'].fillna('word')
# x_test['comment'].fillna('word')
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(x_train['comment'])

# TokenFile = open("TokenFile", "wb")
# pickle.dump(tokenizer, TokenFile)
# TokenFile.close()

train_sequences=tokenizer.texts_to_sequences(x_train['comment'])
test_sequences = tokenizer.texts_to_sequences(x_test['comment'])

train_data = pad_sequences(train_sequences, maxlen=maxlen)
test_data = pad_sequences(test_sequences, maxlen=maxlen)

train_labels = np.asarray(y_train)
test_labels = np.asarray(y_test)

embedding_dim = 50

CNN_model = Sequential([
    layers.Embedding(max_words, embedding_dim), 
    layers.Conv1D(50, 7, activation='relu'), 
    layers.MaxPooling1D(5), 
    layers.Conv1D(50, 7, activation='relu'), 
    layers.GlobalAveragePooling1D(), 
    layers.Dense(20, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

RNN_model = Sequential([
    layers.Embedding(max_words, embedding_dim), 
    layers.Bidirectional(layers.LSTM(64)), 
    layers.Dense(64, activation='relu'), 
    layers.Dense(1, activation='sigmoid')
])

# CNN_model.summary()

RNN_model.compile(optimizer=RMSprop(lr=1e-4), 
                loss='binary_crossentropy', 
                metrics=['acc'])

history = RNN_model.fit(train_data, train_labels, 
                    epochs=10, 
                    batch_size=132, 
                    validation_data=(test_data, test_labels))

# print(RNN_model.predict(test_text))

# CNN_model.save('new_saved_model.h5', save_format='tf')