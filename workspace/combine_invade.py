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
param_file = BASE_PATH / "param_space" / "invade_param_demes_multi_ns.txt"
out_path_base = BASE_PATH / "results_invade"
combined_name = BASE_PATH / "results_invade_combined" / "invade_param_demes_multi_ns.csv"
graph_info_path = BASE_PATH / "results_invade_combined" / "invade_graph_info.csv"

with open(param_file, "r") as f:
    param_sim = f.readlines()
param_sim = [x.strip() for x in param_sim]
param_sim = [x.split(" ") for x in param_sim]

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
    df_grouped["graph_folder"] = graph_folder

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
    # df_grouped["Algebraic connectivity"] = lambda_2
    # df_grouped["transitivity"] = transitivity

    df_all = pd.concat([df_all, df_grouped], ignore_index=True)

df_all = df_all.drop(columns="fixation_time_weighted_sum")
df_all["num_demes"] = df_all["graph_folder"].apply(lambda x: int(x.split("_")[1].split("demes")[1]))
df_all["deme_size"] = df_all["graph_folder"].apply(lambda x: int(x.split("_")[2].split("size")[1]))
df_all["num_edge_added"] = df_all["graph_name"].apply(lambda x: int(x.split("_")[2]))
df_all["beta"] = df_all["graph_name"].apply(lambda x: float(x.split("_")[3].split("m")[-1]))
df_all["graph_rep"] = df_all["graph_name"].apply(lambda x: int(x.split("_")[4]))

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


    G = nx.read_edgelist(BASE_PATH / graph_path, nodetype=int)
    mean_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()

    # Compute the Laplacian matrix
    L = nx.laplacian_matrix(G).toarray()

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(L)

    # Algebraic connectivity is the second smallest eigenvalue
    lambda_2 = sorted(eigenvalues)[1]

    # transitivity
    transitivity = nx.transitivity(G)

    new_row = pd.DataFrame({
        "graph_name": [graph_name],
        "graph_mean_degree": [mean_degree],
        "Algebraic connectivity": [lambda_2],
        "transitivity": [transitivity],
    })
    # df_grouped["graph_mean_degree"] = mean_degree
    # df_grouped["Algebraic connectivity"] = lambda_2
    # df_grouped["transitivity"] = transitivity

    df_graph = pd.concat([df_graph, new_row], ignore_index=True)

df_graph.to_csv(graph_info_path, index=False, sep="\t")
# %%
