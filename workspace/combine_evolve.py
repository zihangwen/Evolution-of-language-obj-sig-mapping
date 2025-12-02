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
import pickle

BASE_PATH = Path("/home/zihangw/EvoComm")

# %%
param_file = BASE_PATH / "param_space" / "evolve_param_demes_multi_ns.txt"
out_path_base = BASE_PATH / "results_evolve"
combined_name = BASE_PATH / "results_evolve_combined" / "evolve_param_demes_multi_ns.csv"

with open(param_file, "r") as f:
    param_sim = f.readlines()
param_sim = [x.strip() for x in param_sim]
param_sim = [x.split(" ") for x in param_sim]

pop_size_select = 100

# %%
df_all = pd.DataFrame()
for param in param_sim:
    model_name = param[2]
    graph_path = param[3]
    num_runs = int(param[4])
    sample_times = int(param[5])
    temperature = float(param[6])

    graph_base = os.path.dirname(graph_path)
    graph_folder = os.path.basename(graph_base)
    graph_name = os.path.basename(graph_path).split(".")[0]

    out_path = out_path_base / model_name / graph_base / graph_name
    if not out_path.exists():
        print(f"Path {out_path} does not exist, skipping.")
        continue

    files = os.listdir(out_path)
    df_list = pd.DataFrame()
    counter = 0
    for file in files:
        file_path = out_path / file
        with open(file_path, "rb") as f:
            logger = pickle.load(f)
        counter += 1
        final_lang = logger["num_languages"][-1]
        final_lang_time = logger["iteration"][np.where(np.array(logger["num_languages"]) == logger["num_languages"][-1])[0][0]]
        max_fitness = logger["max_fitness"][-1]
        max_self_payoff = logger["max_self_payoff"][-1]
        df = pd.DataFrame({
            "num_generations": [num_runs],
            "final_num_lang": [final_lang],
            "final_num_lang_time": [final_lang_time],
            "max_fitness": [max_fitness],
            "max_self_payoff": [max_self_payoff]
        })
        df_list = pd.concat([df_list, df], ignore_index=True)
        df_list["graph_name"] = graph_name
    
    df_grouped = df_list.groupby("graph_name").agg({
        "num_generations": "mean",
        "final_num_lang": "mean",
        "final_num_lang_time": "mean",
        "max_fitness": "mean",
        "max_self_payoff": "mean"
    }).reset_index()

    df_grouped["num_trials"] = counter
    df_grouped["graph_path"] = graph_path
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
    # df_grouped["lambda_2"] = lambda_2
    # df_grouped["transitivity"] = transitivity

    df_all = pd.concat([df_all, df_grouped], ignore_index=True)

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
