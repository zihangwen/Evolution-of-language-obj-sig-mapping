# %%
import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path

# %%
BASE_PATH = Path("/home/zihangw/EvoComm/real_data/")

# %%
graph_list = {}
# %% ----- ----- ----- ----- ----- Hunter gather forest ----- ----- ----- ----- ----- %% #
# Read the file manually
df = pd.read_csv(BASE_PATH / "orgdata" / "hunter-gatherer-data" / "aax5913_data_file_s3.txt", sep=" ", names=["source", "target", "weight"])

# Build the graph
G_unweighted = nx.from_pandas_edgelist(df, "source", "target") # , edge_attr="weight")
# rename_list = [i for i in range(len(list(G_unweighted.nodes)))]
# mapping = dict(zip(list(G_unweighted.nodes), rename_list))
# G_unweighted = nx.relabel_nodes(G_unweighted, mapping)

graph_list["hg_forest"] = {
    "time": 1,
    "num_nodes": G_unweighted.number_of_nodes(),
    "num_edges": G_unweighted.number_of_edges(),
    "avg_degree": np.mean([d for n, d in G_unweighted.degree()]),
    "avg_clustering": nx.average_clustering(G_unweighted),
    "assortativity": nx.degree_assortativity_coefficient(G_unweighted),
    "avg_shortest_path_length": nx.average_shortest_path_length(G_unweighted),
    "diameter": nx.diameter(G_unweighted),
    "modularity": nx.community.quality.modularity(G_unweighted, nx.community.louvain_communities(G_unweighted)),
}

print("Forest: is connected?", nx.is_connected(G_unweighted))
print("Forest: num nodes", graph_list["hg_forest"]["num_nodes"])
print("Forest: num edges", graph_list["hg_forest"]["num_edges"])
print("Forest: avg degree", graph_list["hg_forest"]["avg_degree"])
print("Forest: avg clustering", graph_list["hg_forest"]["avg_clustering"])
print("Forest: assortativity", graph_list["hg_forest"]["assortativity"])
print("Forest: avg shortest path length", graph_list["hg_forest"]["avg_shortest_path_length"])
print("Forest: diameter", graph_list["hg_forest"]["diameter"])
print("Forest: modularity (Louvain)", graph_list["hg_forest"]["modularity"])

# remove self-loops
G_unweighted.remove_edges_from(nx.selfloop_edges(G_unweighted))
nx.write_edgelist(G_unweighted, BASE_PATH / "cleaned" / "hg_forest.txt", data=False)

# %% ----- ----- ----- ----- ----- Hunter gather coastal ----- ----- ----- ----- ----- %% #
# Read the file manually
df = pd.read_csv(BASE_PATH / "orgdata" / "hunter-gatherer-data" / "aax5913_data_file_s4.txt", sep=" ", names=["source", "target", "weight"])

# Build the graph
G_unweighted = nx.from_pandas_edgelist(df, "source", "target") # , edge_attr="weight")
# rename_list = [i for i in range(len(list(G_unweighted.nodes)))]
# mapping = dict(zip(list(G_unweighted.nodes), rename_list))
# G_unweighted = nx.relabel_nodes(G_unweighted, mapping)

graph_list["hg_coastal"] = {
    "time": 1,
    "num_nodes": G_unweighted.number_of_nodes(),
    "num_edges": G_unweighted.number_of_edges(),
    "avg_degree": np.mean([d for n, d in G_unweighted.degree()]),
    "avg_clustering": nx.average_clustering(G_unweighted),
    "assortativity": nx.degree_assortativity_coefficient(G_unweighted),
    "avg_shortest_path_length": nx.average_shortest_path_length(G_unweighted),
    "diameter": nx.diameter(G_unweighted),
    "modularity": nx.community.quality.modularity(G_unweighted, nx.community.louvain_communities(G_unweighted)),
}

print("Coastal: is connected?", nx.is_connected(G_unweighted))
print("Coastal: num nodes", graph_list["hg_coastal"]["num_nodes"])
print("Coastal: num edges", graph_list["hg_coastal"]["num_edges"])
print("Coastal: avg degree", graph_list["hg_coastal"]["avg_degree"])
print("Coastal: avg clustering", graph_list["hg_coastal"]["avg_clustering"])
print("Coastal: assortativity", graph_list["hg_coastal"]["assortativity"])
print("Coastal: avg shortest path length", graph_list["hg_coastal"]["avg_shortest_path_length"])
print("Coastal: diameter", graph_list["hg_coastal"]["diameter"])
print("Coastal: modularity (Louvain)", graph_list["hg_coastal"]["modularity"])

# remove self-loops
G_unweighted.remove_edges_from(nx.selfloop_edges(G_unweighted))
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

graph_list["padgett_florence_families"] = {
    "time": 3,
    "num_nodes": G_unweighted.number_of_nodes(),
    "num_edges": G_unweighted.number_of_edges(),
    "avg_degree": np.mean([d for n, d in G_unweighted.degree()]),
    "avg_clustering": nx.average_clustering(G_unweighted),
    "assortativity": nx.degree_assortativity_coefficient(G_unweighted),
    "avg_shortest_path_length": nx.average_shortest_path_length(G_unweighted),
    "diameter": nx.diameter(G_unweighted),
    "modularity": nx.community.quality.modularity(G_unweighted, nx.community.louvain_communities(G_unweighted)),
}

print("Padgett-Florence-Families: is connected?", nx.is_connected(G_unweighted))
print("Padgett-Florence-Families: num nodes", graph_list["padgett_florence_families"]["num_nodes"])
print("Padgett-Florence-Families: num edges", graph_list["padgett_florence_families"]["num_edges"])
print("Padgett-Florence-Families: avg degree", graph_list["padgett_florence_families"]["avg_degree"])
print("Padgett-Florence-Families: avg clustering", graph_list["padgett_florence_families"]["avg_clustering"])
print("Padgett-Florence-Families: assortativity", graph_list["padgett_florence_families"]["assortativity"])
print("Padgett-Florence-Families: avg shortest path length", graph_list["padgett_florence_families"]["avg_shortest_path_length"])
print("Padgett-Florence-Families: diameter", graph_list["padgett_florence_families"]["diameter"])
print("Padgett-Florence-Families: modularity (Louvain)", graph_list["padgett_florence_families"]["modularity"])

# remove self-loops
G_unweighted.remove_edges_from(nx.selfloop_edges(G_unweighted))
nx.write_edgelist(G_unweighted, BASE_PATH / "cleaned" / "padgett_florence_families.txt", data=False)

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

    if not ((len(list(G_unweighted.nodes))) == max(list(G_unweighted.nodes)) + 1) or (min(list(G_unweighted.nodes)) != 0):
        raise ValueError("Node labels are not starting from 0 to N-1")

    graph_list[f"village_all_{i_v}"] = {
        "time": 2,
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "avg_degree": np.mean([d for n, d in G.degree()]),
        "avg_clustering": nx.average_clustering(G),
        "assortativity": nx.degree_assortativity_coefficient(G),
        "avg_shortest_path_length": nx.average_shortest_path_length(G),
        "diameter": nx.diameter(G),
        "modularity": nx.community.quality.modularity(G, nx.community.louvain_communities(G)),
    }
    
    print(f"Village {village}: is connected?", nx.is_connected(G))
    print(f"Village {village}: num nodes", graph_list[f"village_all_{i_v}"]["num_nodes"])
    print(f"Village {village}: num edges", graph_list[f"village_all_{i_v}"]["num_edges"])
    print(f"Village {village}: avg degree", graph_list[f"village_all_{i_v}"]["avg_degree"])
    print(f"Village {village}: avg clustering", graph_list[f"village_all_{i_v}"]["avg_clustering"])
    print(f"Village {village}: assortativity", graph_list[f"village_all_{i_v}"]["assortativity"])
    print(f"Village {village}: avg shortest path length", graph_list[f"village_all_{i_v}"]["avg_shortest_path_length"])
    print(f"Village {village}: diameter", graph_list[f"village_all_{i_v}"]["diameter"])

    # remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))
    nx.write_edgelist(G, BASE_PATH / "cleaned" / f"village_all_{i_v}.txt", data=False)

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

if not ((len(list(G_unweighted.nodes))) == max(list(G_unweighted.nodes)) + 1) or (min(list(G_unweighted.nodes)) != 0):
    raise ValueError("Node labels are not starting from 0 to N-1")

graph_list["email_eu"] = {
    "time": 4,
    "num_nodes": G_unweighted.number_of_nodes(),
    "num_edges": G_unweighted.number_of_edges(),
    "avg_degree": np.mean([d for n, d in G_unweighted.degree()]),
    "avg_clustering": nx.average_clustering(G_unweighted),
    "assortativity": nx.degree_assortativity_coefficient(G_unweighted),
    "avg_shortest_path_length": nx.average_shortest_path_length(G_unweighted),
    "diameter": nx.diameter(G_unweighted),
    "modularity": nx.community.quality.modularity(G_unweighted, nx.community.louvain_communities(G_unweighted)),
}

print("email-Eu: is connected?", nx.is_connected(G_unweighted))
print("email-Eu: num nodes", graph_list["email_eu"]["num_nodes"])
print("email-Eu: num edges", graph_list["email_eu"]["num_edges"])
print("email-Eu: avg degree", graph_list["email_eu"]["avg_degree"])
print("email-Eu: avg clustering", graph_list["email_eu"]["avg_clustering"])
print("email-Eu: assortativity", graph_list["email_eu"]["assortativity"])
print("email-Eu: avg shortest path length", graph_list["email_eu"]["avg_shortest_path_length"])
print("email-Eu: diameter", graph_list["email_eu"]["diameter"])
print("email-Eu: modularity (Louvain)", graph_list["email_eu"]["modularity"])

# remove self-loops
G_unweighted.remove_edges_from(nx.selfloop_edges(G_unweighted))
nx.write_edgelist(G_unweighted, BASE_PATH / "cleaned" / "email_eu.txt", data=False)

# %% ----- ----- ----- ----- ----- facebook ----- ----- ----- ----- ----- %% #
# # Read the file manually
# df = pd.read_csv(BASE_PATH / "orgdata" / "facebook" / "facebook_combined.txt", sep=" ", names=["nodeID1", "nodeID2"])

# # Build the graph
# G_unweighted = nx.from_pandas_edgelist(df, "nodeID1", "nodeID2") # , edge_attr="weight")
# rename_list = [i for i in range(len(list(G_unweighted.nodes)))]
# mapping = dict(zip(list(G_unweighted.nodes), rename_list))
# G_unweighted = nx.relabel_nodes(G_unweighted, mapping)

# graph_list["facebook"] = {
#     "time": 5,
#     "num_nodes": G_unweighted.number_of_nodes(),
#     "num_edges": G_unweighted.number_of_edges(),
#     "avg_degree": np.mean([d for n, d in G_unweighted.degree()]),
#     "avg_clustering": nx.average_clustering(G_unweighted),
#     "assortativity": nx.degree_assortativity_coefficient(G_unweighted),
#     "avg_shortest_path_length": nx.average_shortest_path_length(G_unweighted),
#     "diameter": nx.diameter(G_unweighted),
#     "modularity": nx.community.quality.modularity(G_unweighted, nx.community.louvain_communities(G_unweighted)),
# }

# print("facebook: is connected?", nx.is_connected(G_unweighted))
# print("facebook: num nodes", graph_list["facebook"]["num_nodes"])
# print("facebook: num edges", graph_list["facebook"]["num_edges"])
# print("facebook: avg degree", graph_list["facebook"]["avg_degree"])
# print("facebook: avg clustering", graph_list["facebook"]["avg_clustering"])
# print("facebook: assortativity", graph_list["facebook"]["assortativity"])
# print("facebook: avg shortest path length", graph_list["facebook"]["avg_shortest_path_length"])
# print("facebook: diameter", graph_list["facebook"]["diameter"])
# print("facebook: modularity (Louvain)", graph_list["facebook"]["modularity"])

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

    # if not nx.is_connected(G_unweighted):
    G_max_connected_list = max(nx.connected_components(G_unweighted), key=len)
    G_max_connected = G_unweighted.subgraph(G_max_connected_list)
    rename_list = [i for i in range(len(list(G_max_connected.nodes)))]
    mapping = dict(zip(list(G_max_connected.nodes), rename_list))
    G_max_connected = nx.relabel_nodes(G_max_connected, mapping)
    G_unweighted = G_max_connected

    if not ((len(list(G_unweighted.nodes))) == max(list(G_unweighted.nodes)) + 1) or (min(list(G_unweighted.nodes)) != 0):
        raise ValueError("Node labels are not starting from 0 to N-1")


    graph_list[f"facebook_{i_fb}"] = {
        "time": 5,
        "num_nodes": G_unweighted.number_of_nodes(),
        "num_edges": G_unweighted.number_of_edges(),
        "avg_degree": np.mean([d for n, d in G_unweighted.degree()]),
        "avg_clustering": nx.average_clustering(G_unweighted),
        "assortativity": nx.degree_assortativity_coefficient(G_unweighted),
        "avg_shortest_path_length": nx.average_shortest_path_length(G_unweighted),
        "diameter": nx.diameter(G_unweighted),
        "modularity": nx.community.quality.modularity(G_unweighted, nx.community.louvain_communities(G_unweighted)),
    }

    print(f"facebook {i_fb}: is connected?", nx.is_connected(G_unweighted))
    print(f"facebook {i_fb}: num nodes", graph_list[f"facebook_{i_fb}"]["num_nodes"])
    print(f"facebook {i_fb}: num edges", graph_list[f"facebook_{i_fb}"]["num_edges"])
    print(f"facebook {i_fb}: avg degree", graph_list[f"facebook_{i_fb}"]["avg_degree"])
    print(f"facebook {i_fb}: avg clustering", graph_list[f"facebook_{i_fb}"]["avg_clustering"])
    print(f"facebook {i_fb}: assortativity", graph_list[f"facebook_{i_fb}"]["assortativity"])
    print(f"facebook {i_fb}: avg shortest path length", graph_list[f"facebook_{i_fb}"]["avg_shortest_path_length"])
    print(f"facebook {i_fb}: diameter", graph_list[f"facebook_{i_fb}"]["diameter"])
    print(f"facebook {i_fb}: modularity (Louvain)", graph_list[f"facebook_{i_fb}"]["modularity"])

    # remove self-loops
    G_unweighted.remove_edges_from(nx.selfloop_edges(G_unweighted))
    nx.write_edgelist(G_unweighted, BASE_PATH / "cleaned" / f"facebook_{i_fb}.txt", data=False)

# %% ----- ----- ----- ----- ----- retweet copen ----- ----- ----- ----- ----- %% #
# G = nx.read_edgelist(BASE_PATH / "orgdata" / "retweet" / "rt-twitter-copen.mtx")
df = pd.read_csv(BASE_PATH / "orgdata" / "retweet" / "rt-twitter-copen.mtx", sep=" ", skiprows=2, names=["nodeID1", "nodeID2"])

# Build the graph
G_unweighted = nx.from_pandas_edgelist(df, "nodeID1", "nodeID2") # , edge_attr="weight")
# if not nx.is_connected(G_unweighted):
G_max_connected_list = max(nx.connected_components(G_unweighted), key=len)
G_max_connected = G_unweighted.subgraph(G_max_connected_list)
rename_list = [i for i in range(len(list(G_max_connected.nodes)))]
mapping = dict(zip(list(G_max_connected.nodes), rename_list))
G_max_connected = nx.relabel_nodes(G_max_connected, mapping)
G_unweighted = G_max_connected

if not ((len(list(G_unweighted.nodes))) == max(list(G_unweighted.nodes)) + 1) or (min(list(G_unweighted.nodes)) != 0):
    raise ValueError("Node labels are not starting from 0 to N-1")

graph_list["retweet_copen"] = {
    "time": 6,
    "num_nodes": G_unweighted.number_of_nodes(),
    "num_edges": G_unweighted.number_of_edges(),
    "avg_degree": np.mean([d for n, d in G_unweighted.degree()]),
    "avg_clustering": nx.average_clustering(G_unweighted),
    "assortativity": nx.degree_assortativity_coefficient(G_unweighted),
    "avg_shortest_path_length": nx.average_shortest_path_length(G_unweighted),
    "diameter": nx.diameter(G_unweighted),
    "modularity": nx.community.quality.modularity(G_unweighted, nx.community.louvain_communities(G_unweighted)),
}

print("retweet copen: is connected?", nx.is_connected(G_unweighted))
print("retweet copen: num nodes", graph_list["retweet_copen"]["num_nodes"])
print("retweet copen: num edges", graph_list["retweet_copen"]["num_edges"])
print("retweet copen: avg degree", graph_list["retweet_copen"]["avg_degree"])
print("retweet copen: avg clustering", graph_list["retweet_copen"]["avg_clustering"])
print("retweet copen: assortativity", graph_list["retweet_copen"]["assortativity"])
print("retweet copen: avg shortest path length", graph_list["retweet_copen"]["avg_shortest_path_length"])
print("retweet copen: diameter", graph_list["retweet_copen"]["diameter"])
print("retweet copen: modularity (Louvain)", graph_list["retweet_copen"]["modularity"])

# remove self-loops
G_unweighted.remove_edges_from(nx.selfloop_edges(G_unweighted))
nx.write_edgelist(G_max_connected, BASE_PATH / "cleaned" / "retweet_copen.txt", data=False)

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
