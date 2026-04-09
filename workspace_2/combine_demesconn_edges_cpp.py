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
param_file = BASE_PATH / "param_space" / "invade_param_demesconn_edges_supp.txt"
out_path_base = BASE_PATH / "results_invade_cpp"
combined_name = BASE_PATH / "results_cpp_combined" / "param_demesconn_edges_invade_supp.csv"

param_file_e = BASE_PATH / "param_space" / "evolve_param_demes_multi_ns.txt"
out_path_base_e = BASE_PATH / "results_evolve_cpp"
combined_name_e = BASE_PATH / "results_cpp_combined" / "param_demesconn_edges_evolve_supp.csv"

graph_info_path = BASE_PATH / "results_cpp_combined" / "param_demesconn_edges_info_supp.csv"

with open(param_file, "r") as f:
    param_sim = f.readlines()
param_sim = [x.strip() for x in param_sim]
param_sim = [x.split(" ") for x in param_sim]

with open(param_file_e, "r") as f:
    param_sim_e = f.readlines()
param_sim_e = [x.strip() for x in param_sim_e]
param_sim_e = [x.split(" ") for x in param_sim_e]

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

    files = os.listdir(out_path)
    df_list = [pd.read_csv(out_path / file, sep="\t") for file in files]
    df = pd.concat(df_list, ignore_index=True)
    df = df.rename(columns={df.columns[0]: df.columns[0][2:]})

    df["fixation_time_weighted_sum"] = df["fixation_time"] * df["fixation_count"]
    if "graph_name:" in df.columns:
        df = df.rename(columns={"graph_name:": "graph_name"})
    
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
    df_grouped["graph_name"] = graph_name
    df_grouped["graph_path"] = graph_path

    df_all = pd.concat([df_all, df_grouped], ignore_index=True)

df_all = df_all.drop(columns="fixation_time_weighted_sum")
combined_name.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
df_all.to_csv(combined_name, index=False, sep="\t")

# %%
df_all = pd.DataFrame()
for param in param_sim_e:
    model_name = param[2]
    graph_path = param[3]
    num_runs = int(param[4])
    sample_times = int(param[5])
    # temperature = float(param[6])

    graph_base = os.path.dirname(graph_path)
    graph_folder = os.path.basename(graph_base)
    graph_name = os.path.basename(graph_path).split(".")[0]

    out_path = out_path_base_e / model_name / graph_base / graph_name
    if not out_path.exists():
        print(f"Path {out_path} does not exist, skipping.")
        continue

    files = os.listdir(out_path)
    df_list = pd.DataFrame()
    counter = 0
    for file in files:
        file_path = out_path / file
        df_temp = pd.read_csv(file_path, sep="\t")
        logger = df_temp.to_dict(orient="list")
        
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
    df_grouped["graph_name"] = graph_name
    df_grouped["graph_path"] = graph_path

    df_all = pd.concat([df_all, df_grouped], ignore_index=True)

combined_name_e.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
df_all.to_csv(combined_name_e, index=False, sep="\t")

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

    if "wm" in graph_name:
        num_demes = 1
        demeconn_number = 0
        num_edge_added = 0
        # beta = 0
        pop_size = int(graph_base.split("_")[-1])
        deme_size = pop_size
        rep = 0
    elif "star" in graph_name:
        num_demes = int(graph_name.split("ndeme")[-1].split("_")[0])
        demeconn_number = "star"
        num_edge_added = int(graph_name.split("edge")[-1].split("_")[0])
        # beta = int(graph_name.split("beta")[-1].split("_")[0])
        pop_size = int(graph_base.split("pop")[-1].split("_")[0])
        deme_size = pop_size // num_demes
        rep = int(graph_name.split("rep")[-1])
    else:
        num_demes = int(graph_name.split("ndeme")[-1].split("_")[0])
        demeconn_number = int(graph_name.split("demeconn")[-1].split("_")[0])
        num_edge_added = int(graph_name.split("edge")[-1].split("_")[0])
        # beta = int(graph_name.split("beta")[-1].split("_")[0])
        pop_size = int(graph_base.split("pop")[-1].split("_")[0])
        deme_size = pop_size // num_demes
        rep = int(graph_name.split("rep")[-1])
    
    # if pop_size != pop_size_select:
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
        "graph_path": [graph_path],
        "graph_name": [graph_name],
        "num_demes": [num_demes],
        "demeconn_number": [demeconn_number],
        "deme_size": [deme_size],
        "num_edge_added": [num_edge_added],
        # "beta": [beta],
        "pop_size": [pop_size],
        "rep": [rep],
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

    df_graph = pd.concat([df_graph, new_row], ignore_index=True)

graph_info_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
df_graph.to_csv(graph_info_path, index=False, sep="\t")

# %%
