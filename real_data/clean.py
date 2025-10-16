# %%
import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path

# %%
BASE_PATH = Path("/home/zihangw/EvoComm/real_data/")

# %% ----- ----- ----- ----- ----- Hunter gather forest ----- ----- ----- ----- ----- %% #
# Read the file manually
df = pd.read_csv(BASE_PATH / "orgdata" / "hunter-gatherer-data" / "aax5913_data_file_s3.txt", sep=" ", names=["source", "target", "weight"])

# Build the graph
G_unweighted = nx.from_pandas_edgelist(df, "source", "target") # , edge_attr="weight")
rename_list = [i for i in range(len(list(G_unweighted.nodes)))]
mapping = dict(zip(list(G_unweighted.nodes), rename_list))
G_unweighted = nx.relabel_nodes(G_unweighted, mapping)

print("Forest: is connected?", nx.is_connected(G_unweighted))
print("Forest: num nodes", G_unweighted.number_of_nodes())
print("Forest: num edges", G_unweighted.number_of_edges())
print("Forest: avg degree", np.mean([d for n, d in G_unweighted.degree()]))
print("Forest: avg clustering", nx.average_clustering(G_unweighted))
print("Forest: assortativity", nx.degree_assortativity_coefficient(G_unweighted))
print("Forest: avg shortest path length", nx.average_shortest_path_length(G_unweighted))
print("Forest: diameter", nx.diameter(G_unweighted))
print("Forest: modularity (Louvain)", nx.community.quality.modularity(G_unweighted, nx.community.louvain_communities(G_unweighted)))

nx.write_edgelist(G_unweighted, BASE_PATH / "cleaned" / "hg_forest.txt", data=False)

# %% ----- ----- ----- ----- ----- Hunter gather coastal ----- ----- ----- ----- ----- %% #
# Read the file manually
df = pd.read_csv(BASE_PATH / "orgdata" / "hunter-gatherer-data" / "aax5913_data_file_s4.txt", sep=" ", names=["source", "target", "weight"])

# Build the graph
G_unweighted = nx.from_pandas_edgelist(df, "source", "target") # , edge_attr="weight")
rename_list = [i for i in range(len(list(G_unweighted.nodes)))]
mapping = dict(zip(list(G_unweighted.nodes), rename_list))
G_unweighted = nx.relabel_nodes(G_unweighted, mapping)

print("Coastal: is connected?", nx.is_connected(G_unweighted))
print("Coastal: num nodes", G_unweighted.number_of_nodes())
print("Coastal: num edges", G_unweighted.number_of_edges())
print("Coastal: avg degree", np.mean([d for n, d in G_unweighted.degree()]))
print("Coastal: avg clustering", nx.average_clustering(G_unweighted))
print("Coastal: assortativity", nx.degree_assortativity_coefficient(G_unweighted))
print("Coastal: avg shortest path length", nx.average_shortest_path_length(G_unweighted))
print("Coastal: diameter", nx.diameter(G_unweighted))
print("Coastal: modularity (Louvain)", nx.community.quality.modularity(G_unweighted, nx.community.louvain_communities(G_unweighted)))

nx.write_edgelist(G_unweighted, BASE_PATH / "cleaned" / "hg_coastal.txt", data=False)

# %% ----- ----- ----- ----- ----- Padgett-Florence-Families ----- ----- ----- ----- ----- %% #
# # Read the file manually
df = pd.read_csv(BASE_PATH / "orgdata" / "Padgett-Florence-Families_Multiplex_Social" / "Dataset" / "Padgett-Florentine-Families_multiplex.edges", sep=" ", names=["layerID", "nodeID1", "nodeID2", "weight"])

# We only use layerID 1 (marriage) and 2 (business)
# df = df[df["layerID"].isin([1, 2])]

# Build the graph
G_unweighted = nx.from_pandas_edgelist(df, "nodeID1", "nodeID2") # , edge_attr="weight")
rename_list = [i for i in range(len(list(G_unweighted.nodes)))]
mapping = dict(zip(list(G_unweighted.nodes), rename_list))
G_unweighted = nx.relabel_nodes(G_unweighted, mapping)

print("Padgett-Florence-Families: is connected?", nx.is_connected(G_unweighted))
print("Padgett-Florence-Families: num nodes", G_unweighted.number_of_nodes())
print("Padgett-Florence-Families: num edges", G_unweighted.number_of_edges())
print("Padgett-Florence-Families: avg degree", np.mean([d for n, d in G_unweighted.degree()]))
print("Padgett-Florence-Families: avg clustering", nx.average_clustering(G_unweighted))
print("Padgett-Florence-Families: assortativity", nx.degree_assortativity_coefficient(G_unweighted))
print("Padgett-Florence-Families: avg shortest path length", nx.average_shortest_path_length(G_unweighted))
print("Padgett-Florence-Families: diameter", nx.diameter(G_unweighted))
print("Padgett-Florence-Families: modularity (Louvain)", nx.community.quality.modularity(G_unweighted, nx.community.louvain_communities(G_unweighted)))

# nx.write_edgelist(G_unweighted, BASE_PATH / "cleaned" / "padgett_florence_families.txt", data=False)

# %% ----- ----- ----- ----- ----- Banerjee et al. Indian villages all ----- ----- ----- ----- ----- %% #
# Read the file manually
village_list = [
    (v, f"adj_allVillageRelationships_vilno_{v}.csv")
    for v in range(1, 78) if v not in [13, 22]
]
# village = village_list[0]

for i_v, village in village_list:
    A = np.loadtxt(BASE_PATH / "orgdata"/ "village" / "datav4.0" / "Data" / "1. Network Data" / "Adjacency Matrices" / f"{village}", delimiter=",")
    print(A.shape)

    G = nx.from_numpy_array(A)

    if not nx.is_connected(G):
        G_max_connected_list = max(nx.connected_components(G), key=len)
        G_max_connected = G.subgraph(G_max_connected_list)
        rename_list = [i for i in range(len(list(G_max_connected.nodes)))]
        mapping = dict(zip(list(G_max_connected.nodes), rename_list))
        G_max_connected = nx.relabel_nodes(G_max_connected, mapping)
        G = G_max_connected

    print(f"Village {village}: is connected?", nx.is_connected(G))
    print(f"Village {village}: num nodes", G.number_of_nodes())
    print(f"Village {village}: num edges", G.number_of_edges())
    print(f"Village {village}: avg degree", np.mean([d for n, d in G.degree()]))
    print(f"Village {village}: avg clustering", nx.average_clustering(G))
    print(f"Village {village}: assortativity", nx.degree_assortativity_coefficient(G))
    print(f"Village {village}: avg shortest path length", nx.average_shortest_path_length(G))
    print(f"Village {village}: diameter", nx.diameter(G))
    print(f"Village {village}: modularity (Louvain)", nx.community.quality.modularity(G, nx.community.louvain_communities(G)))

    # nx.write_edgelist(G, BASE_PATH / "cleaned" / f"village_all_{i_v}.txt", data=False)

# %% ----- ----- ----- ----- ----- Banerjee et al. Indian villages AND ----- ----- ----- ----- ----- %% #
# village_list = [
#     f"adj_andRelationships_vilno_{v}.csv"
#     for v in range(1, 78) if v not in [13, 22]
# ]
# # village = village_list[0]

# for i_v, village in enumerate(village_list):
#     A = np.loadtxt(BASE_PATH / "orgdata"/ "village" / "datav4.0" / "Data" / "1. Network Data" / "Adjacency Matrices" / f"{village}", delimiter=",")
#     print(A.shape)

#     G = nx.from_numpy_array(A)

#     if not nx.is_connected(G):
#         G_max_connected_list = max(nx.connected_components(G), key=len)
#         G_max_connected = G.subgraph(G_max_connected_list)
#         rename_list = [i for i in range(len(list(G_max_connected.nodes)))]
#         mapping = dict(zip(list(G_max_connected.nodes), rename_list))
#         G_max_connected = nx.relabel_nodes(G_max_connected, mapping)
#         G = G_max_connected

#     print(f"Village {village}: is connected?", nx.is_connected(G))
#     print(f"Village {village}: num nodes", G.number_of_nodes())
#     print(f"Village {village}: num edges", G.number_of_edges())
#     print(f"Village {village}: avg degree", np.mean([d for n, d in G.degree()]))
#     print(f"Village {village}: avg clustering", nx.average_clustering(G))
#     print(f"Village {village}: assortativity", nx.degree_assortativity_coefficient(G))
#     print(f"Village {village}: avg shortest path length", nx.average_shortest_path_length(G))
#     print(f"Village {village}: diameter", nx.diameter(G))
#     print(f"Village {village}: modularity (Louvain)", nx.community.quality.modularity(G, nx.community.louvain_communities(G)))

#     nx.write_edgelist(G, BASE_PATH / "cleaned" / f"village_and_{i_v}.txt", data=False)

# %% ----- ----- ----- ----- ----- email-Eu ----- ----- ----- ----- ----- %% #
# Read the file manually
df = pd.read_csv(BASE_PATH / "orgdata" / "email-Eu" / "email-Eu-core.txt", sep=" ", names=["nodeID1", "nodeID2"])

# We only use layerID 1 (marriage) and 2 (business)
# df = df[df["layerID"].isin([1, 2])]

# Build the graph
G_unweighted = nx.from_pandas_edgelist(df, "nodeID1", "nodeID2") # , edge_attr="weight")
if not nx.is_connected(G_unweighted):
    G_max_connected_list = max(nx.connected_components(G_unweighted), key=len)
    G_max_connected = G_unweighted.subgraph(G_max_connected_list)
    rename_list = [i for i in range(len(list(G_max_connected.nodes)))]
    mapping = dict(zip(list(G_max_connected.nodes), rename_list))
    G_max_connected = nx.relabel_nodes(G_max_connected, mapping)
    G_unweighted = G_max_connected

print("email-Eu: is connected?", nx.is_connected(G_unweighted))
print("email-Eu: num nodes", G_unweighted.number_of_nodes())
print("email-Eu: num edges", G_unweighted.number_of_edges())
print("email-Eu: avg degree", np.mean([d for n, d in G_unweighted.degree()]))
print("email-Eu: avg clustering", nx.average_clustering(G_unweighted))
print("email-Eu: assortativity", nx.degree_assortativity_coefficient(G_unweighted))
print("email-Eu: avg shortest path length", nx.average_shortest_path_length(G_unweighted))
print("email-Eu: diameter", nx.diameter(G_unweighted))
print("email-Eu: modularity (Louvain)", nx.community.quality.modularity(G_unweighted, nx.community.louvain_communities(G_unweighted)))

# nx.write_edgelist(G_unweighted, BASE_PATH / "cleaned" / "email_eu.txt", data=False)

# %% ----- ----- ----- ----- ----- facebook ----- ----- ----- ----- ----- %% #
# # Read the file manually
df = pd.read_csv(BASE_PATH / "orgdata" / "facebook" / "facebook_combined.txt", sep=" ", names=["nodeID1", "nodeID2"])

# Build the graph
G_unweighted = nx.from_pandas_edgelist(df, "nodeID1", "nodeID2") # , edge_attr="weight")
rename_list = [i for i in range(len(list(G_unweighted.nodes)))]
mapping = dict(zip(list(G_unweighted.nodes), rename_list))
G_unweighted = nx.relabel_nodes(G_unweighted, mapping)


print("facebook: is connected?", nx.is_connected(G_unweighted))
print("facebook: num nodes", G_unweighted.number_of_nodes())
print("facebook: num edges", G_unweighted.number_of_edges())
print("facebook: avg degree", np.mean([d for n, d in G_unweighted.degree()]))
print("facebook: avg clustering", nx.average_clustering(G_unweighted))
print("facebook: assortativity", nx.degree_assortativity_coefficient(G_unweighted))
print("facebook: avg shortest path length", nx.average_shortest_path_length(G_unweighted))
print("facebook: diameter", nx.diameter(G_unweighted))
print("facebook: modularity (Louvain)", nx.community.quality.modularity(G_unweighted, nx.community.louvain_communities(G_unweighted)))

# nx.write_edgelist(G_unweighted, BASE_PATH / "cleaned" / "facebook.txt", data=False)

# %% ----- ----- ----- ----- ----- facebook separate ----- ----- ----- ----- ----- %% #
fb_separate_list = [
    (fb, f"{fb}.edges")
    for fb in [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]
]

for i_fb, fb_separate in fb_separate_list:
    df = pd.read_csv(BASE_PATH / "orgdata" / "facebook" / "facebook" / f"{fb_separate}", sep=" ", names=["nodeID1", "nodeID2"])

    # Build the graph
    G_unweighted = nx.from_pandas_edgelist(df, "nodeID1", "nodeID2") # , edge_attr="weight")

    if not nx.is_connected(G_unweighted):
        G_max_connected_list = max(nx.connected_components(G_unweighted), key=len)
        G_max_connected = G_unweighted.subgraph(G_max_connected_list)
        rename_list = [i for i in range(len(list(G_max_connected.nodes)))]
        mapping = dict(zip(list(G_max_connected.nodes), rename_list))
        G_max_connected = nx.relabel_nodes(G_max_connected, mapping)
        G_unweighted = G_max_connected

    print(f"facebook {fb_separate}: is connected?", nx.is_connected(G_unweighted))
    print(f"facebook {fb_separate}: num nodes", G_unweighted.number_of_nodes())
    print(f"facebook {fb_separate}: num edges", G_unweighted.number_of_edges())
    print(f"facebook {fb_separate}: avg degree", np.mean([d for n, d in G_unweighted.degree()]))
    print(f"facebook {fb_separate}: avg clustering", nx.average_clustering(G_unweighted))
    print(f"facebook {fb_separate}: assortativity", nx.degree_assortativity_coefficient(G_unweighted))
    print(f"facebook {fb_separate}: avg shortest path length", nx.average_shortest_path_length(G_unweighted))
    print(f"facebook {fb_separate}: diameter", nx.diameter(G_unweighted))
    print(f"facebook {fb_separate}: modularity (Louvain)", nx.community.quality.modularity(G_unweighted, nx.community.louvain_communities(G_unweighted)))

    # nx.write_edgelist(G_unweighted, BASE_PATH / "cleaned" / f"facebook_{i_fb}.txt", data=False)

# %% ----- ----- ----- ----- ----- retweet copen ----- ----- ----- ----- ----- %% #
# G = nx.read_edgelist(BASE_PATH / "orgdata" / "retweet" / "rt-twitter-copen.mtx")
df = pd.read_csv(BASE_PATH / "orgdata" / "retweet" / "rt-twitter-copen.mtx", sep=" ", skiprows=2, names=["nodeID1", "nodeID2"])

# Build the graph
G_unweighted = nx.from_pandas_edgelist(df, "nodeID1", "nodeID2") # , edge_attr="weight")
if not nx.is_connected(G_unweighted):
    G_max_connected_list = max(nx.connected_components(G_unweighted), key=len)
    G_max_connected = G_unweighted.subgraph(G_max_connected_list)
    rename_list = [i for i in range(len(list(G_max_connected.nodes)))]
    mapping = dict(zip(list(G_max_connected.nodes), rename_list))
    G_max_connected = nx.relabel_nodes(G_max_connected, mapping)
    G_unweighted = G_max_connected

print("retweet copen: is connected?", nx.is_connected(G_max_connected))
print("retweet copen: num nodes", G_max_connected.number_of_nodes())
print("retweet copen: num edges", G_max_connected.number_of_edges())
print("retweet copen: avg degree", np.mean([d for n, d in G_max_connected.degree()]))
print("retweet copen: avg clustering", nx.average_clustering(G_max_connected))
print("retweet copen: assortativity", nx.degree_assortativity_coefficient(G_max_connected))
print("retweet copen: avg shortest path length", nx.average_shortest_path_length(G_max_connected))
print("retweet copen: diameter", nx.diameter(G_max_connected))
print("retweet copen: modularity (Louvain)", nx.community.quality.modularity(G_max_connected, nx.community.louvain_communities(G_max_connected)))

# nx.write_edgelist(G_max_connected, BASE_PATH / "cleaned" / "retweet_copen.txt", data=False)

# %%















# %%
# def load_G_weighted(f):
#     G = dict()
#     el = np.loadtxt(f).astype(int)
#     for edge in el:
#         n1, n2, weight = edge
#         if n1 in G:
#             G[n1] += [n2]
#         else:
#             G[n1] = [n2]

#         if n2 in G:
#             G[n2] += [n1]
#         else:
#             G[n2] = [n1]
#     return G

# forest_dict = load_G_weighted("aax5913_data_file_s3.txt")
# coastal_dict = load_G_weighted("aax5913_data_file_s4.txt")
