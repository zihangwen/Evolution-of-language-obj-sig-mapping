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

# sys.path.append(str(Path("/home/zihangw/EvoComm/code")))
# from simulations import Logger

BASE_PATH = Path("/home/zihangw/EvoComm")
plt.rcParams.update({'font.size': 20})

# %%
param_file = BASE_PATH / "param_space" / "evolve_param_demes_multi_ns.txt"
out_path_base = BASE_PATH / "results_evolve"
# combined_name = BASE_PATH / "results_invade_combined" / "invade_param_demes_multi.csv"
# graph_info_path = BASE_PATH / "results_invade_combined" / "invade_graph_info.csv"

with open(param_file, "r") as f:
    param_sim = f.readlines()
param_sim = [x.strip() for x in param_sim]
param_sim = [x.split(" ") for x in param_sim]

data_all = defaultdict(list)
for param in param_sim:
    # num_objects = int(param[0])
    # num_sounds = int(param[1])
    model_name = param[2]
    graph_path = param[3]
    num_runs = int(param[4])
    sample_times = int(param[5])
    temperature = float(param[6])

    graph_base = os.path.dirname(graph_path)
    graph_base_name = graph_base.split("/")[-1]
    graph_name = os.path.basename(graph_path).split(".")[0]
    out_path = out_path_base / graph_base / graph_name
    
    num_demes = int(graph_name.split("_")[1])
    num_edge_added = int(graph_name.split("_")[2])
    beta = int(graph_name.split("_")[3].split("m")[-1])
    deme_size = int(graph_base.split("size")[-1])

    final_lang = []
    final_lang_time = []
    counter = 0
    for i in range(100):
        graph_base = os.path.dirname(graph_path)
        graph_name = os.path.basename(graph_path).split(".")[0]

        out_path = os.path.join(out_path_base, model_name, graph_base, graph_name)
        file_path = os.path.join(out_path, f"st_{sample_times}_temp_{temperature:.1f}_{i}.pkl")

        try:
            with open(file_path, "rb") as f:
                logger = pickle.load(f)
            counter += 1
        except FileNotFoundError:
            continue

        # df = pd.DataFrame(logger)
        final_lang = logger["num_languages"][-1]
        final_lang_time = logger["iteration"][np.where(np.array(logger["num_languages"]) == logger["num_languages"][-1])[0][0]]
    
    data_all["graph_name"].append(graph_base_name)
    data_all["num_trials"].append(counter)
    data_all["num_demes"].append(num_demes)
    data_all["deme_size"].append(deme_size)
    data_all["num_edge_added"].append(num_edge_added)
    data_all["beta"].append(beta)
    data_all["num_generations"].append(num_runs)

    data_all["final_num_lang"].append(np.mean(final_lang).item())
    data_all["final_num_lang_time"].append(np.mean(final_lang_time).item())

# %%
df = pd.DataFrame(data_all)
deme_size_list = df['deme_size'].unique()

# %%
fig, axes = plt.subplots(1, len(deme_size_list), figsize=(12 * len(deme_size_list), 8), sharex='col')

# plt.figure(figsize=(12, 8))
for i, deme_size in enumerate(deme_size_list):
    ax = axes[i]
    df_select = df[df['deme_size'] == deme_size]
    sns.lineplot(data=df_select, x="num_edge_added", y="final_num_lang", hue="num_demes", markers=True, dashes=False, ax=ax)
    ax.set_title(f"Deme Size: {deme_size}")
    ax.set_xlabel("Number of Edges Added")
    ax.set_ylabel("Final Number of Languages")

# %%
fig, axes = plt.subplots(1, len(deme_size_list), figsize=(12 * len(deme_size_list), 8), sharex='col')

# plt.figure(figsize=(12, 8))
for i, deme_size in enumerate(deme_size_list):
    ax = axes[i]
    df_select = df[df['deme_size'] == deme_size]
    sns.lineplot(data=df_select, x="num_edge_added", y="final_num_lang_time", hue="num_demes", markers=True, dashes=False, ax=ax)
    ax.set_title(f"Deme Size: {deme_size}")
    ax.set_xlabel("Number of Edges Added")
    ax.set_ylabel("Time to Reach Final Number of Languages")

# %%
