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
param_file = BASE_PATH / "param_space" / "invade_param_demes_fix_popsize.txt"
out_path_base = BASE_PATH / "results_invade"
combined_name = BASE_PATH / "results_invade_combined" / "invade_param_demes_fix_popsize.csv"
graph_info_path = BASE_PATH / "results_invade_combined" / "invade_param_demes_fix_popsize_info.csv"

with open(param_file, "r") as f:
    param_sim = f.readlines()
param_sim = [x.strip() for x in param_sim]
param_sim = [x.split(" ") for x in param_sim]

pop_size_select = 100

# %%
df_all = pd.DataFrame()
for param in param_sim:
    graph_path = param[2]
    graph_base = os.path.dirname(graph_path)
    graph_folder = os.path.basename(graph_base)
    graph_name = os.path.basename(graph_path).split(".")[0]
    out_path = out_path_base / graph_base / graph_name
    if not out_path.exists():
        print(f"Path {out_path} does not exist, skipping.")
        continue

    if "wm" in graph_name:
        num_demes = 1
        num_edge_added = 0
        beta = 0
        pop_size = int(graph_base.split("_")[-1])
        deme_size = pop_size
        rep = 0
    else:
        num_demes = int(graph_name.split("ndeme")[-1].split("_")[0])
        num_edge_added = int(graph_name.split("edge")[-1].split("_")[0])
        beta = int(graph_name.split("beta")[-1].split("_")[0])
        pop_size = int(graph_base.split("pop")[-1])
        deme_size = pop_size // num_demes
        rep = int(graph_name.split("rep")[-1])
    
    if pop_size != pop_size_select:
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
        if row["fixation_count"] > 0 else 0,
        axis=1
    )

    df_grouped["pfix"] = df_grouped["fixation_count"] / df_grouped["num_trials"]
    df_grouped["pco_exist"] = df_grouped["co_existence_count"] / df_grouped["num_trials"]
    # df_grouped["graph_folder"] = graph_folder

    df_grouped["num_demes"] = num_demes
    df_grouped["deme_size"] = deme_size
    df_grouped["num_edge_added"] = num_edge_added
    df_grouped["beta"] = beta
    # df_grouped["pop_size"] = pop_size
    df_grouped["graph_name"] = graph_name

    df_grouped["graph_base"] = graph_folder.split("_")[0]

    # G = nx.read_edgelist(BASE_PATH / graph_path, nodetype=int)
    # mean_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()

    # # Compute the Laplacian matrix
    # L = nx.laplacian_matrix(G).toarray()

    # # Compute eigenvalues
    # eigenvalues = np.linalg.eigvalsh(L)

    # # Algebraic connectivity is the second smallest eigenvalue
    # lambda_2 = sorted(eigenvalues)[1]

    # # transitivity
    # transitivity = nx.transitivity(G)

    # df_grouped["graph_mean_degree"] = mean_degree
    # df_grouped["lambda_2"] = lambda_2
    # df_grouped["transitivity"] = transitivity

    df_all = pd.concat([df_all, df_grouped], ignore_index=True)

# df_all = df_all.drop(columns="fixation_time_weighted_sum")
# df_all["num_demes"] = df_all["graph_folder"].apply(lambda x: int(x.split("_")[1].split("demes")[1]))
# df_all["deme_size"] = df_all["graph_folder"].apply(lambda x: int(x.split("_")[2].split("size")[1]))
# df_all["num_edge_added"] = df_all["graph_name"].apply(lambda x: int(x.split("_")[2]))
# df_all["beta"] = df_all["graph_name"].apply(lambda x: float(x.split("_")[3].split("m")[-1]))
# df_all["graph_rep"] = df_all["graph_name"].apply(lambda x: int(x.split("_")[4]))

df_all = df_all.drop(columns="fixation_time_weighted_sum")
# df_all["num_demes"] = df_all["graph_folder"].apply(lambda x: int(x.split("_")[1].split("demes")[1]))
# df_all["deme_size"] = df_all["graph_folder"].apply(lambda x: int(x.split("_")[2].split("size")[1]))
# df_all["num_edge_added"] = df_all["graph_name"].apply(lambda x: int(x.split("_")[2]))
# df_all["beta"] = df_all["graph_name"].apply(lambda x: float(x.split("_")[3].split("m")[-1]))
# df_all["graph_rep"] = df_all["graph_name"].apply(lambda x: int(x.split("_")[4]))

combined_name.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
df_all.to_csv(combined_name, index=False, sep="\t")

    # for file in files:
    #     if file.endswith(".txt"):
    #         file_path = out_path / file
    #         with open(file_path, "r") as f:
    #             lines = f.readlines()
    #         lines = [x.strip() for x in lines]
    #         lines = [x.split("\t") for x in lines]
    #         if len(lines) > 1:
    #             header = lines[0]
    #             data = np.array(lines[1:])
    #             data = data[:, 1:].astype(float)
    #             data_mean = np.mean(data, axis=0)
    #             data_std = np.std(data, axis=0)
    #             print(f"{graph_name} {file} mean: {data_mean}, std: {data_std}")

    
# %%
# df_graph = pd.DataFrame()
# for param in param_sim:
#     print(param)
#     graph_path = param[2]
#     graph_base = os.path.dirname(graph_path)
#     graph_name = os.path.basename(graph_path).split(".")[0]
#     out_path = out_path_base / graph_base / graph_name
#     if not out_path.exists():
#         print(f"Path {out_path} does not exist, skipping.")
#         continue

#     G = nx.read_edgelist(BASE_PATH / graph_path, nodetype=int)
#     mean_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()

#     # Compute the Laplacian matrix
#     L = nx.laplacian_matrix(G).toarray()

#     # Compute eigenvalues
#     eigenvalues = np.linalg.eigvalsh(L)

#     # Algebraic connectivity is the second smallest eigenvalue
#     lambda_2 = sorted(eigenvalues)[1]

#     # transitivity
#     transitivity = nx.transitivity(G)

#     new_row = pd.DataFrame({
#         "graph_name": [graph_name],
#         "graph_mean_degree": [mean_degree],
#         "Algebraic connectivity": [lambda_2],
#         "transitivity": [transitivity],
#     })
#     # df_grouped["graph_mean_degree"] = mean_degree
#     # df_grouped["Algebraic connectivity"] = lambda_2
#     # df_grouped["transitivity"] = transitivity

#     df_graph = pd.concat([df_graph, new_row], ignore_index=True)

# df_graph.to_csv(graph_info_path, index=False, sep="\t")

# %%
df_graph = pd.DataFrame()
for param in param_sim:
    print(param)
    graph_path = param[2]
    graph_base = os.path.dirname(graph_path)
    graph_name = os.path.basename(graph_path).split(".")[0]
    # out_path = out_path_base / graph_base / graph_name
    # if not out_path.exists():
    #     print(f"Path {out_path} does not exist, skipping.")
    #     continue

    G = nx.read_edgelist(BASE_PATH / graph_path, nodetype=int)
    pop_size = G.number_of_nodes()
    if pop_size != pop_size_select:
        continue

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
