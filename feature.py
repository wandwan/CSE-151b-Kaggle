import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox

# Read the data
chunk = pd.read_csv('train.csv', chunksize=1000, nrows=200)

# Concatenate the chunks
df = pd.concat(chunk)
id_to_index = {}
index = 0
# call type, origin call, origin stand, taxi id, timestamp.
taxi_ids = set()
# make one hot encoding for taxi id
taxi_id_onehot = np.zeros((df['TAXI_ID'].nunique(), df.shape[0]), dtype=np.bool)
for i in range(1, df.shape[0]):
    if df['TAXI_ID'][i] not in id_to_index:
        id_to_index[df['TAXI_ID'][i]] = index
        index += 1
    taxi_id_onehot[id_to_index[df['TAXI_ID'][i]]][i] = True
taxi_id_onehot = np.packbits(taxi_id_onehot, axis=0)
index = 0
# make one hot encoding for call type
call_type_onehot = np.zeros((df['CALL_TYPE'].nunique(), df.shape[0]), dtype=np.bool)
for i in range(1, df.shape[0]):
    if df['CALL_TYPE'][i] not in id_to_index:
        id_to_index[df['CALL_TYPE'][i]] = index
        index += 1
    call_type_onehot[id_to_index[df['CALL_TYPE'][i]]][i] = True
call_type_onehot = np.packbits(call_type_onehot, axis=0)

# make timestamp encoding (percent of year, percent of day, percent of week), one hot for week, one hot for hour
timestamp = np.zeros((df.shape[0], 4), dtype=np.float32)
time_one_hots = np.zeros((df.shape[0], 31), dtype=np.bool)
for i in range(1, df.shape[0]):
    timestamp[i][0] = (df['TIMESTAMP'][i] / 60**2 / 24 / 365.0) % 1
    timestamp[i][1] = (df['TIMESTAMP'][i] / 60**2 / 24 / 7.0) % 1
    timestamp[i][2] = (df['TIMESTAMP'][i] / 60**2 / 24.0) % 1
    timestamp[i][3] = (df['TIMESTAMP'][i] / 60.0**2) % 1
    time_one_hots[i][int(timestamp[i][1] * 7)] = True
    time_one_hots[i][int(timestamp[i][3] * 24) + 7] = True

output = []
for polyline in df['POLYLINE']:
    polyline = [tuple(map(float, coord.split(',')))
                for coord in polyline[2:-2].split('],[')]
    output.append(len(polyline) * 15)
output = np.array(output, dtype=np.int32)

# Save the data
np.savez_compressed('train.npz', taxi_id_onehot=taxi_id_onehot, call_type_onehot=call_type_onehot,
                    time_one_hots=time_one_hots, output=output)
