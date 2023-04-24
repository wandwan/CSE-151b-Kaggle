from functools import reduce
import pandas as pd
import numpy as np
import math
import ast

# Define the chunksize
chunksize = 1000
ORIG_COORD = [-8.60789730,41.14766290]

# Read the data in chunks
for i, chunk in enumerate(pd.read_csv('train.csv', chunksize=chunksize)):
    df = chunk.loc[chunk['MISSING_DATA'] == False, 'POLYLINE']
    # Split each row of the POLYLINE column into overlapping sequences of length 8
    result, ans = [], []
    for row in df:
      row = [ORIG_COORD, *ast.literal_eval(row)]
      if len(row) <= 2:
        continue
      if len(row) < 9:
        row = [*row, *([row[-1]] * (9 - len(row)))]
      for i in range(len(row) - 1, 0, -1):
        x,x1 = row[i - 1], row[i]
        row[i] = (math.atan2(x1[1] - x[1], x1[0] - x[0]), 
                  math.sqrt((x1[1] - x[1])**2 + (x1[0] - x[0])**2))
      rowAns = [*row[9:], (0,0)]
      for i in range(1, len(row) - 7):
        row[i] = row[i:i + 8]
      row = row[1:-7]
      result += row
      ans += rowAns
    arr = np.array(result)
    out = np.array(ans)
    # Save the numpy array to a file
    print(f'Saving chunk {i} with shape {arr.shape}')
    print(f'Saving chunk {i} with shape {out.shape}')
    np.save(f'train_chunk_{i}.npy', arr)
    np.save(f'train_ans_{i}.npy', out)
