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

# Prepare the data
# Read the data
chunk = pd.read_csv('train.csv', chunksize=1000, nrows=200)

# Concatenate the chunkspolyline
df = pd.concat(chunk)
df = df[(df.MISSING_DATA == False)]
df = df["POLYLINE"]

# Define a function that converts a string to a numpy array
def string_to_array(s):
  # Use ast.literal_eval to evaluate the string as a list of lists
  lst = ast.literal_eval(s)
  # Use np.array to convert the list of lists to a numpy array
  arr = np.array(lst)
  # Return the numpy array
  return arr

# Define a function that splits a coordinate list into overlapping sequences of length 8
def split_list(lst):
  # Initialize an empty list to store the sequences
  seqs = []
  # Loop through the list with a step size of 4
  for i in range(0, len(lst) - 7, 1):
    # Slice the list from i to i + 8 and append it to the seqs list
    seqs.append(lst[i:i + 8])
  # Return the seqs list as a numpy array
  return np.array(seqs)

# Apply the string_to_array function to the column A and assign the result to a new column called B
df["B"] = df["A"].apply(string_to_array)

# Convert the column B to a numpy array of shape (n, m, 2) where n is the number of rows and m is the length of each coordinate list
arr = np.stack(df["B"].values)

# Apply the split_list function to each row of arr and concatenate the results along axis 0
result = np.concatenate([split_list(row) for row in arr], axis=0)

print(result.shape)
print(result)