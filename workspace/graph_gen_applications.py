# %%
import networkx as nx
import numpy as np
import os
import copy
import random

base_path = "/home/zihangw/EvoComm"
# %% social dolphins and social caltech36
# folder = "soc-dolphins"
# name = "soc-dolphins"
# num_lines_ignored = 36

# folder = "socfb-Caltech36"
# name = "socfb-Caltech36"
# num_lines_ignored = 2

folder = "retweet"
name = "rt-twitter-copen"
num_lines_ignored = 0

with open(os.path.join(base_path, f"networks_archieve/social_network_org/{folder}/{name}.mtx"), "r") as f:
    lines = f.readlines()[num_lines_ignored:]  # Skip the first two lines

# Use nx.parse_edgelist instead of nx.read_edgelist
G = nx.parse_edgelist(lines, nodetype=int)

G_max_connected_list = max(nx.connected_components(G), key=len)
G_max_connected = G.subgraph(G_max_connected_list)
rename_list = [i for i in range(len(list(G_max_connected.nodes)))]
mapping = dict(zip(list(G_max_connected.nodes), rename_list))
G_max_connected = nx.relabel_nodes(G_max_connected, mapping)
print(nx.is_connected(G_max_connected))
print(len(G_max_connected.nodes()))
degree = np.array(G.degree())[:,1].astype(int)
N = G.number_of_nodes()
mean_degree = np.mean(degree)
var_degree = np.var(degree)
print(mean_degree)
nx.write_edgelist(
    G_max_connected,
    os.path.join(base_path, f"networks/social_network/{name}-max.txt"),
    data=False
)

# %% retweet graph bahrain
folder = "retweet"
name = "rt_bahrain"

with open(os.path.join(base_path, f"networks_archieve/social_network_org/{folder}/{name}.edges"), "r") as rf:
    lines = [line.rstrip() for line in rf]
lines_write = [line.split(",")[0] + " " + line.split(",")[1] for line in lines]

G = nx.parse_edgelist(lines_write, nodetype=int)

G_max_connected_list = max(nx.connected_components(G), key=len)
G_max_connected = G.subgraph(G_max_connected_list)
rename_list = [i for i in range(len(list(G_max_connected.nodes)))]
mapping = dict(zip(list(G_max_connected.nodes), rename_list))  
G_max_connected = nx.relabel_nodes(G_max_connected, mapping)
print(nx.is_connected(G_max_connected))
nx.write_edgelist(
    G_max_connected,
    os.path.join(base_path, f"networks/social_network/{name}-max.txt"),
    data=False
)

