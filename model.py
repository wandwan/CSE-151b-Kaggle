import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
import ast


# LSTM to predict the angle + magnitude of the next vector in our sequence
model = keras.Sequential()
model.add(layers.LSTM(32, input_shape=(8, 2), return_sequences=True))
model.add(layers.LSTM(32, return_sequences=True))
model.add(layers.LSTM(8))
model.add(layers.Dense(2, activation='relu'))

def dist_loss(y_true, y_pred):
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=1)))

model.compile(loss=dist_loss, optimizer='adam', metrics=['accuracy'])

for i in range(100):
    path, ans = np.load(f'train_chunk_{i}.npy'), np.load(f'train_ans_{i}.npy')
    model.fit(path, ans, epochs=1, batch_size=32, verbose=1)
