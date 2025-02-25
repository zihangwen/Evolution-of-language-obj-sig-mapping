# %%
import networkx as nx
import numpy as np
import os
import copy
import random
from itertools import pairwise


# Island: N, d, mig_rate
# Bottleneck
base_path = "/home/zihangw/EvoComm/"

# %%
# ----- ----- ----- ----- ----- bottleneck models ----- ----- ----- ----- ----- #
n = 100
num_dims = 10
# num_edges_list = [1, 2, 5, 10, 20, 50]
# beta_list = [-3, -2, -1, 0, 1, 2]
num_edges_list = [1, 5, 20]
beta_list = [-3, 0, 2]

output_path = os.path.join(base_path, "networks", "bottleneck_demes")
os.makedirs(output_path,exist_ok=True)

dim_size = n // num_dims
G_1 = nx.complete_graph(dim_size)

G_base = nx.empty_graph()
node_list = []
G_list = []
dim_index_list = []
for i_dim in range(num_dims):
    G_dim = copy.deepcopy(G_1)
    mapping = {node: node + i_dim * dim_size for node in G_dim.nodes}
    G_dim = nx.relabel_nodes(G_dim, mapping)
    
    G_base = nx.compose(G_base, G_dim)
    node_list.append(G_dim.nodes)
    G_list.append(G_dim)
    dim_index_list.append(i_dim)

# %%
pairs = list(pairwise(dim_index_list))
for num_edges in num_edges_list:
    for beta in beta_list:
        count_m = 0
        G = copy.deepcopy(G_base)
        for i_edge in range(num_edges):
            degree_list = [np.array(G_i.degree())[:,1].astype(float) for G_i in G_list]
            prob_list = [degree ** beta for degree in degree_list]
            prob_list = [prob / np.sum(prob) for prob in prob_list]

            new_edge_list = []
            for dim_1, dim_2 in pairs:
                node_1 = np.random.choice(node_list[dim_1], 1, False, prob_list[dim_1])[0]
                node_2 = np.random.choice(node_list[dim_2], 1, False, prob_list[dim_2])[0]
                new_edge_list.append((node_1, node_2))
            
            # print(new_edge_list)
            G.add_edges_from(new_edge_list)
        print(nx.is_connected(G))
        nx.write_edgelist(G, os.path.join(output_path,f"bn_{num_dims}_{num_edges}_m{str(beta)}_{str(count_m)}.txt"), data=False)
            

# os.listdir("/home/zihangw/EvoComm/networks/bottleneck")