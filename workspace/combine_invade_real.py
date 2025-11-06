# %%
import numpy as np
import pandas as pd
from dataclasses import dataclass
import copy
import time
import os
import sys
from pathlib import Path
import networkx as nx

BASE_PATH = Path("/home/zihangw/EvoComm")

# %%
param_file = BASE_PATH / "param_space" / "invade_real_clean.txt"
out_path_base = BASE_PATH / "results_invade"
combined_name = BASE_PATH / "results_invade_combined" / "invade_real_clean.csv"
graph_info_path = BASE_PATH / "results_invade_combined" / "invade_real_clean_info.csv"

with open(param_file, "r") as f:
    param_sim = f.readlines()
param_sim = [x.strip() for x in param_sim]
param_sim = [x.split(" ") for x in param_sim]

# pop_size_select = 100
# %%
df_all = pd.DataFrame()
for param in param_sim:
    graph_path = param[2]
    graph_base = os.path.dirname(graph_path)
    # graph_folder = os.path.basename(graph_base)
    graph_name = os.path.basename(graph_path).split(".")[0]
    out_path = out_path_base / graph_base / graph_name
    if not out_path.exists():
        print(f"Path {out_path} does not exist, skipping.")
        continue

    files = os.listdir(out_path)
    df_list = [pd.read_csv(out_path / file, sep="\t") for file in files]
    df = pd.concat(df_list, ignore_index=True)
    df = df.rename(columns={df.columns[0]: df.columns[0][2:]})

    df["fixation_time_weighted_sum"] = df["fixation_time"] * df["fixation_count"]
    
    df_grouped = df.groupby("graph_name").agg({
        "num_trials": "sum",
        "fixation_count": "sum",
        "fixation_time_weighted_sum": "sum",
        "co_existence_count": "sum"
    }).reset_index()

    df_grouped["fixation_time"] = df_grouped.apply(
        lambda row: row["fixation_time_weighted_sum"] / row["fixation_count"]
        if row["fixation_count"] > 0 else np.inf,
        axis=1
    )

    df_grouped["pfix"] = df_grouped["fixation_count"] / df_grouped["num_trials"]
    df_grouped["pco_exist"] = df_grouped["co_existence_count"] / df_grouped["num_trials"]
    # df_grouped["graph_folder"] = graph_folder

    df_grouped["graph_name"] = graph_name

    df_all = pd.concat([df_all, df_grouped], ignore_index=True)

df_all = df_all.drop(columns="fixation_time_weighted_sum")

combined_name.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
df_all.to_csv(combined_name, index=False, sep="\t")


# %%
df_graph = pd.DataFrame()
for param in param_sim:
    print(param)
    graph_path = param[2]
    graph_base = os.path.dirname(graph_path)
    graph_name = os.path.basename(graph_path).split(".")[0]
    out_path = out_path_base / graph_base / graph_name
    if not out_path.exists():
        print(f"Path {out_path} does not exist, skipping.")
        continue

    G = nx.read_edgelist(BASE_PATH / "real_data" / graph_path, nodetype=int)

    mean_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    average_clustering = nx.average_clustering(G)
    assortativity = nx.degree_assortativity_coefficient(G)
    diameter = nx.diameter(G)
    modularity = nx.community.quality.modularity(G, nx.community.louvain_communities(G))
    transitivity = nx.transitivity(G)
    connectivity = nx.algebraic_connectivity(G)
    new_row = pd.DataFrame({
        "graph_name": [graph_name],
        "num_nodes": [G.number_of_nodes()],
        "num_edges": [G.number_of_edges()],
        "graph_mean_degree": [mean_degree],
        "average_clustering": [average_clustering],
        "assortativity": [assortativity],
        "diameter": [diameter],
        "modularity": [modularity],
        "transitivity": [transitivity],
        "algebraic_connectivity": [connectivity]
    })
    # new_row["num_nodes"] = G.number_of_nodes()
    # new_row["num_edges"] = G.number_of_edges()
    # new_row["graph_mean_degree"] = mean_degree
    # new_row["average_clustering"] = average_clustering
    # new_row["assortativity"] = assortativity
    # new_row["diameter"] = diameter
    # new_row["modularity"] = modularity

    # # Algebraic connectivity is the second smallest eigenvalue
    # L = nx.laplacian_matrix(G).toarray()
    # eigenvalues = np.linalg.eigvalsh(L)
    # lambda_2 = sorted(eigenvalues)[1]

    # new_row["graph_mean_degree"] = mean_degree
    # new_row["lambda_2"] = lambda_2

    df_graph = pd.concat([df_graph, new_row], ignore_index=True)

graph_info_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
df_graph.to_csv(graph_info_path, index=False, sep="\t")

# %%
