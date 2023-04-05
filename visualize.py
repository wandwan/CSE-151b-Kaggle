import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox

# Read the data
chunk = pd.read_csv('train.csv', chunksize=1000, nrows=200)

# Concatenate the chunks
df = pd.concat(chunk)

# Get some Base Information
print(df.head())
print(df.keys())

# Get the number of unique values for each column
for col in df.keys():
    if col != 'POLYLINE':
        print(df[col].value_counts())

# Plot each polyline in red on a road network from Porto, Portugal
G = ox.graph_from_place('Porto, Portugal', network_type='drive')
all = []
for polyline in df['POLYLINE']:
    polyline = [tuple(map(float, coord.split(',')))
                for coord in polyline[2:-2].split('],[')]
    polyline = [ox.nearest_nodes(G, coord[0], coord[1])
                for coord in polyline]
    all.append(polyline)
#     fixed_polyline = []
#     for i in range(len(polyline) - 1):
#         nodelist = ox.shortest_path(G, polyline[i], polyline[i + 1])
#         if nodelist is not None:
#             if len(fixed_polyline) > 0 and fixed_polyline[-1] == nodelist[0]:
#                 fixed_polyline.extend(nodelist[1:])
#             else:
#                 fixed_polyline.extend(nodelist)
#         else:
#             if len(fixed_polyline) > 0:
#                 all.append(fixed_polyline)
#                 fixed_polyline = []
#     if len(fixed_polyline) > 0:
#         all.append(polyline)
# print(all)
# print([len(route) for route in all])
for route in all:
    for u, v in zip(route[:-1], route[1:]):
        if G.has_edge(u, v) is False:
            G.add_edge(u, v, length=1)
ox.plot_graph_routes(G, all, route_color='r',
                     route_linewidth=1, node_size=0)
