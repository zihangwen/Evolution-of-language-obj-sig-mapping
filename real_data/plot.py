# %%
import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
import os

# %%
BASE_PATH = Path("/home/zihangw/EvoComm/real_data/")
graph_list = {}

# %% ----- ----- ----- ----- ----- Hunter gather forest ----- ----- ----- ----- ----- %% #
G_unweighted = nx.read_edgelist(BASE_PATH / "cleaned" / "hg_forest.txt")

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

# plot the graph
nx.draw(G_unweighted, node_size=20, edge_color="gray")


# %% ----- ----- ----- ----- ----- Hunter gather coastal ----- ----- ----- ----- ----- %% #
G_unweighted2 = nx.read_edgelist(BASE_PATH / "cleaned" / "hg_coastal.txt")

graph_list["hg_coastal"] = {
    "time": 1,
    "num_nodes": G_unweighted2.number_of_nodes(),
    "num_edges": G_unweighted2.number_of_edges(),
    "avg_degree": np.mean([d for n, d in G_unweighted2.degree()]),
    "avg_clustering": nx.average_clustering(G_unweighted2),
    "assortativity": nx.degree_assortativity_coefficient(G_unweighted2),
    "avg_shortest_path_length": nx.average_shortest_path_length(G_unweighted2),
    "diameter": nx.diameter(G_unweighted2),
    "modularity": nx.community.quality.modularity(G_unweighted2, nx.community.louvain_communities(G_unweighted2)),
}

print("Coastal: is connected?", nx.is_connected(G_unweighted2))
print("Coastal: num nodes", graph_list["hg_coastal"]["num_nodes"])
print("Coastal: num edges", graph_list["hg_coastal"]["num_edges"])
print("Coastal: avg degree", graph_list["hg_coastal"]["avg_degree"])
print("Coastal: avg clustering", graph_list["hg_coastal"]["avg_clustering"])
print("Coastal: assortativity", graph_list["hg_coastal"]["assortativity"])
print("Coastal: avg shortest path length", graph_list["hg_coastal"]["avg_shortest_path_length"])
print("Coastal: diameter", graph_list["hg_coastal"]["diameter"])
print("Coastal: modularity (Louvain)", graph_list["hg_coastal"]["modularity"])

# plot the graph
nx.draw(G_unweighted2, node_size=20, edge_color="gray")

# %%
