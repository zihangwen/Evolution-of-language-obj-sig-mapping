import networkx as nx
import numpy as np
import os
import copy
import random


base_path = "/home/zihangw/EvoComm/"
# ----- ----- ----- ----- ----- manual model ----- ----- ----- ----- ----- #
# output_path = os.path.join(base_path, "graphs")
# os.makedirs(output_path,exist_ok=True)
# G = nx.random_regular_graph(4, 10)
# while not nx.is_connected(G):
#     G = nx.random_regular_graph(4, 10)

# G_copy = copy.deepcopy(G)
# mapping = {node: node + max(G.nodes) + 1 for node in G.nodes}
# G_copy = nx.relabel_nodes(G_copy, mapping)
# G_combined = nx.compose(G, G_copy)
# inter_edges = [(0, 10), (1, 11), (2, 12), (3, 13)]
# G_combined.add_edges_from(inter_edges)

# nx.write_edgelist(G_combined, os.path.join(output_path,"two_deme_20.txt"), data=False)

# ----- ----- ----- ----- ----- toy model ----- ----- ----- ----- ----- #
# n = 10
# output_path = os.path.join(base_path, "networks", "toy")
# os.makedirs(output_path,exist_ok=True)

# G = nx.complete_graph(n)
# nx.write_edgelist(G, os.path.join(output_path,"wm_" + str(n) + ".txt"), data=False)

# for d in [2,3,4]:
# # for d in [2]:
#     G = nx.random_regular_graph(d, n)
#     while not nx.is_connected(G):
#         G = nx.random_regular_graph(d, n)
#     nx.write_edgelist(G, os.path.join(output_path,"regular_" + str(n) + "_" + str(d) + ".txt"), data=False)

# for d in [2,3,4]:
#     complete_g = n - d
#     G = nx.complete_graph(complete_g)
#     detour_edge_list = [(complete_g - 1 + i_d, complete_g + i_d) for i_d in range(d)]
#     detour_edge_list += [(n-1, 0)]
#     G.add_edges_from(detour_edge_list)
#     nx.write_edgelist(G, os.path.join(output_path,"detour_" + str(n) + "_" + str(d) + ".txt"), data=False)

# G = nx.star_graph(n-1)
# nx.write_edgelist(G, os.path.join(output_path,'star_' + str(n) + '.txt'), data=False)

n = 10
output_path = os.path.join(base_path, "networks", "toy_star")
os.makedirs(output_path,exist_ok=True)

G = nx.star_graph(n-1)
candidate = [(i, j) for i in range(10) for j in range(i + 1, 10) if (i,j) not in G.edges]
# nx.write_edgelist(G, os.path.join(output_path,'star_' + str(n) + '.txt'), data=False)

for i in range(30):
    G_copy = copy.deepcopy(G)
    # num_edges = int((i // 5 + 1) * 5)
    num_edges = int((i // 5 + 1))
    print(num_edges)
    random.shuffle(candidate)
    G_copy.add_edges_from(candidate[:num_edges])
    nx.write_edgelist(G_copy, os.path.join(output_path,'star_' + str(n) + '_' + str(i) + '.txt'), data=False)


# os.listdir("/home/zihangw/EvoComm/networks/toy")
# ----- ----- ----- ----- ----- PA model ----- ----- ----- ----- ----- #
# n = 100
# output_path = os.path.join(base_path, "networks", "PA_100")
# os.makedirs(output_path,exist_ok=True)
# for m in [2, 10, 15, 20, 25]:
#     count_m = 0
#     com_m = 2*m+1
#     for beta in [float(a) for a in [-3, -2, -1, 0, 1, 2]]:
#         G = nx.complete_graph(com_m)
#         for i_node in range(com_m, n):
#             degree = np.array(G.degree())[:,1].astype(int)
#             prob = degree ** beta
#             if len(prob) == 1:
#                 prob = [1]
#             else:
#                 prob /= np.sum(prob)
#             connect_node_chosen = np.random.choice(i_node, m, False, prob)
#             new_edge_list = [(node_chosen, i_node) for node_chosen in connect_node_chosen]
#             G.add_edges_from(new_edge_list)
#         nx.write_edgelist(G, os.path.join(output_path,"PA_" + str(n) + "_m" + str(m) + "_" + str(count_m) + ".txt"), data=False)
#         count_m += 1

# os.listdir("/home/zihangw/EvoComm/networks/PA_100")
