import pandas as pd
import numpy as np
import ast

# Define a function that converts a string to a numpy array
def string_to_array(s):
    # Use ast.literal_eval to evaluate the string as a list of lists
    lst = ast.literal_eval(s)
    # Use np.array to convert the list of lists to a numpy array
    arr = np.array(lst)
    # Return the numpy array
    return arr

def split_list(lst, debug=False):
  # Initialize an empty list to store the sequences
  seqs = []
  # If the length of lst is less than 8, pad with [0,0]
  if len(lst) < 8:
    seq = [list(lst)]
    while len(seq) < 8:
      seq.append([0, 0])
    seq = np.array(seq)
    if debug:
       print(seq)
    seqs.append(seq)
  else:
    if debug:
      print("seqs length", len(lst))
    # Loop through the list with a step size of 4
    for i in range(0, len(lst) - 7, 1):
      # Slice the list from i to i + 8 and append it to the seqs list
      seqs.append(lst[i:i + 8])
  return seqs


# Define the chunksize
chunksize = 100

# Read the data in chunks
for i, chunk in enumerate(pd.read_csv('train.csv', chunksize=chunksize)):

    # Filter out missing data and select the POLYLINE column
    df = chunk.loc[chunk['MISSING_DATA'] == False, 'POLYLINE']

    # Apply the string_to_array function to the POLYLINE column
    df = df.map(string_to_array)
    
    # Filter out rows that have a length of 0
    df = df[df.apply(lambda x: x.shape[0] > 0)]

    # Split each row of the POLYLINE column into overlapping sequences of length 8
    prev = None
    i = 0
    for row in df:
       print("added row", i)
       i += 1
       if i == 55:
          print("debugging")
          print(row)
          result = split_list(row, debug=True)
       else:
          result = split_list(row)
       print("result length", len(result))
       if prev is not None:
          np.append(prev, np.concatenate(result, axis=0), axis=0)
       else:
          prev = np.concatenate(result, axis=0)

    # Convert the resulting sequences to a numpy array of shape (n, m, 2) where n is the number of sequences and m is the length of each sequence (which is 8 in this case)
    arr = np.stack(result)

    # Save the numpy array to a file
    np.save(f'train_chunk_{i}.npy', arr)
