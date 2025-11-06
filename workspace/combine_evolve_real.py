# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path
import os
import pickle
import sys
from collections import defaultdict
import networkx as nx

# sys.path.append(str(Path("/home/zihangw/EvoComm/code")))
# from simulations import Logger

BASE_PATH = Path("/home/zihangw/EvoComm")

# %%
param_file = BASE_PATH / "param_space" / "evolve_real_clean.txt"
out_path_base = BASE_PATH / "results_evolve"
combined_name = BASE_PATH / "results_evolve_combined" / "evolve_param_demes_multi.csv"
# graph_info_path = BASE_PATH / "results_evolve_combined" / "evolve_graph_info.csv"

with open(param_file, "r") as f:
    param_sim = f.readlines()
param_sim = [x.strip() for x in param_sim]
param_sim = [x.split(" ") for x in param_sim]

# %%
df_all = pd.DataFrame()
for param in param_sim:
    # num_objects = int(param[0])
    # num_sounds = int(param[1])
    model_name = param[2]
    graph_path = param[3]
    num_runs = int(param[4])
    sample_times = int(param[5])
    temperature = float(param[6])

    graph_base = os.path.dirname(graph_path)
    # graph_folder = os.path.basename(graph_base)
    graph_name = os.path.basename(graph_path).split(".")[0]

    out_path = out_path_base / model_name / graph_base / graph_name
    
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

    df_all = pd.concat([df_all, df_grouped], ignore_index=True)


combined_name.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
df_all.to_csv(combined_name, index=False, sep="\t")


# %%
