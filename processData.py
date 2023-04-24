import pandas as pd
import numpy as np
import math
import ast

# Define a function that converts a string to a numpy array
def string_to_array(s):
    # Use ast.literal_eval to evaluate the string as a list of lists
    lst = ast.literal_eval(s)
    # Use np.array to convert the list of lists to a numpy array
    arr = np.array(lst)
    # Return the numpy array
    return arr

def split_list(lst):
  lst = [(math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2),
          math.atan2(y[1] - x[1], y[0] - x[0])) for x, y in lst]
  lst.insert(0, (0, 0))
  print(lst)
  ans = []
  # Initialize an empty list to store the sequences
  seqs = []
  # If the length of lst is less than 8, pad with [0,0]
  if len(lst) < 8:
    seq = list(lst)
    while len(seq) < 8:
      seq.append(np.array([0, 0]))
    seq = np.array(seq)
    seqs.append(seq)
    ans.append(np.array([0,0]))
  else:
    # Loop through the list with a step size of 4
    for i in range(0, len(lst) - 7, 4):
      # Slice the list from i to i + 8 and append it to the seqs list
      seqs.append(lst[i:i + 8])
      if(i + 8 >= len(lst)):
        ans.append(np.array([0,0]))
      else:
        ans.append(lst[i + 8])
  return seqs, ans


# Define the chunksize
chunksize = 1000

# Read the data in chunks
for i, chunk in enumerate(pd.read_csv('train.csv', chunksize=chunksize)):
    # Filter out missing data and select the POLYLINE column
    df = chunk.loc[chunk['MISSING_DATA'] == False, 'POLYLINE']
    # Split each row of the POLYLINE column into overlapping sequences of length 8
    arr = None
    out = None
    for row in df:
      row = string_to_array(row)
      if len(row) == 0:
        continue
      result, ans = [np.stack(x, axis=0) for x in split_list(row)]
      if arr is not None:
        arr = np.append(arr, result, axis=0)
      else:
        arr = result
      if out is not None:
        out = np.append(out, ans, axis=0)
      else:
        out = ans
    
    # Save the numpy array to a file
    print(f'Saving chunk {i} with shape {arr.shape}')
    print(f'Saving chunk {i} with shape {out.shape}')
    np.save(f'train_chunk_{i}.npy', arr)
    np.save(f'train_ans_{i}.npy', out)
