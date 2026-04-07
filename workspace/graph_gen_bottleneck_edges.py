# %%
import networkx as nx
import numpy as np
import os
import copy
import random
from itertools import pairwise
from pathlib import Path


# Island: N, d, mig_rate
# Bottleneck
base_path = Path("/home/zihangw/EvoComm/")

# %%
# ----- ----- ----- ----- ----- complete graph pop size ----- ----- ----- ----- ----- #
# n = 500
# G_wm = nx.complete_graph(n)
# output_path = base_path / "networks" / f"wm_{n}"
# os.makedirs(output_path, exist_ok=True)
# nx.write_edgelist(G_wm, output_path / f"wm_{n}.txt", data=False)

# %%
# ----- ----- ----- ----- ----- bottleneck models fix pop size ----- ----- ----- ----- ----- #
n = 100
# num_demes_list = [5, 10, 20]
# between_deme_conn = {
#     5: [2, 4],
#     10: [2, 5, 9],
#     20: [2, 5, 10, 15, 19]
# }
num_demes_list = [10, 20]
between_deme_conn = {
    10: [4],
    20: [4]
}
num_edges_list = [1, 2, 5, 10]
# beta_list = [0]

# n = 500
# num_demes_list = [5, 10, 20, 50, 100]
# num_edges_list = [1, 2, 5, 10, 20]
# beta_list = [0]

# n = 1000
# num_demes_list = [5, 10, 20, 40, 50, 100]
# num_edges_list = [1, 2, 5, 10, 20, 40]
# beta_list = [0]

output_path = base_path / "networks" / f"bottleneck_pop{n}_demeconn"
os.makedirs(output_path, exist_ok=True)

# %%
for num_demes in num_demes_list:
    deme_size = n // num_demes
    G_1 = nx.complete_graph(deme_size)

    G_base = nx.empty_graph()
    node_list = []
    G_list = []
    dim_index_list = []
    for i_dim in range(num_demes):
        G_dim = copy.deepcopy(G_1)
        mapping = {node: node + i_dim * deme_size for node in G_dim.nodes}
        G_dim = nx.relabel_nodes(G_dim, mapping)
        
        G_base = nx.compose(G_base, G_dim)
        node_list.append(G_dim.nodes)
        G_list.append(G_dim)
        dim_index_list.append(i_dim)
    
    for conn in between_deme_conn[num_demes]:
        while True:
            graph_connection_template = nx.random_regular_graph(conn, num_demes)
            if nx.is_connected(graph_connection_template):
                break
        pairs = list(graph_connection_template.edges)
        # pairs = list(pairwise(dim_index_list))
        for num_edges in num_edges_list:
            # for beta in beta_list:
            beta = 0
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
            nx.write_edgelist(G, os.path.join(output_path,f"bn_ndeme{num_demes}_demeconn{conn}_edge{num_edges}_rep{str(count_m)}.txt"), data=False)

# %% diy graphs
for num_demes in num_demes_list:
    deme_size = n // num_demes
    G_1 = nx.complete_graph(deme_size)

    G_base = nx.empty_graph()
    node_list = []
    G_list = []
    dim_index_list = []
    for i_dim in range(num_demes):
        G_dim = copy.deepcopy(G_1)
        mapping = {node: node + i_dim * deme_size for node in G_dim.nodes}
        G_dim = nx.relabel_nodes(G_dim, mapping)
        
        G_base = nx.compose(G_base, G_dim)
        node_list.append(G_dim.nodes)
        G_list.append(G_dim)
        dim_index_list.append(i_dim)
    
    for diy_graph in ["star", ]:
        while True:
            if diy_graph == "star":
                conn = num_demes - 1
                graph_connection_template = nx.star_graph(num_demes - 1)
            if nx.is_connected(graph_connection_template):
                break
        pairs = list(graph_connection_template.edges)
        # pairs = list(pairwise(dim_index_list))
        for num_edges in num_edges_list:
            # for beta in beta_list:
            beta = 0
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
            nx.write_edgelist(G, os.path.join(output_path,f"bn_ndeme{num_demes}_{diy_graph}_edge{num_edges}_rep{str(count_m)}.txt"), data=False)

# %%
# ----- ----- ----- ----- ----- bottleneck models ----- ----- ----- ----- ----- #
# deme_size_list = [5, 10, 20]
# # num_demes_list = [5, 10, 20, 50]
# num_demes_list = [1]
# # num_demes = 10
# # num_edges_list = [1, 2, 5, 10, 20, 50]
# # beta_list = [-3, -2, -1, 0, 1, 2]
# # num_edges_list = [5, 10, 20, 40]
# num_edges_list = [0]
# beta_list = [0]

# for deme_size in deme_size_list:
#     for num_demes in num_demes_list:
#         output_path = base_path / "networks" / f"bottleneck_demes{num_demes}_size{deme_size}"
#         os.makedirs(output_path,exist_ok=True)

#         G_1 = nx.complete_graph(deme_size)

#         G_base = nx.empty_graph()
#         node_list = []
#         G_list = []
#         dim_index_list = []
#         for i_dim in range(num_demes):
#             G_dim = copy.deepcopy(G_1)
#             mapping = {node: node + i_dim * deme_size for node in G_dim.nodes}
#             G_dim = nx.relabel_nodes(G_dim, mapping)
            
#             G_base = nx.compose(G_base, G_dim)
#             node_list.append(G_dim.nodes)
#             G_list.append(G_dim)
#             dim_index_list.append(i_dim)

#         pairs = list(pairwise(dim_index_list))
#         for num_edges in num_edges_list:
#             for beta in beta_list:
#                 count_m = 0
#                 G = copy.deepcopy(G_base)
#                 for i_edge in range(num_edges):
#                     degree_list = [np.array(G_i.degree())[:,1].astype(float) for G_i in G_list]
#                     prob_list = [degree ** beta for degree in degree_list]
#                     prob_list = [prob / np.sum(prob) for prob in prob_list]

#                     new_edge_list = []
#                     for dim_1, dim_2 in pairs:
#                         node_1 = np.random.choice(node_list[dim_1], 1, False, prob_list[dim_1])[0]
#                         node_2 = np.random.choice(node_list[dim_2], 1, False, prob_list[dim_2])[0]
#                         new_edge_list.append((node_1, node_2))
                    
#                     # print(new_edge_list)
#                     G.add_edges_from(new_edge_list)
#                 print(nx.is_connected(G))
#                 nx.write_edgelist(G, os.path.join(output_path, f"bn_{num_demes}_{num_edges}_m{str(beta)}_{str(count_m)}.txt"), data=False)

# os.listdir("/home/zihangw/EvoComm/networks/bottleneck")
# %% test
# G = nx.read_edgelist("/home/zihangw/EvoComm/networks/bottleneck_demes_100/bn_100_5_m0_0.txt", nodetype=int, data=False)
# G.nodes
