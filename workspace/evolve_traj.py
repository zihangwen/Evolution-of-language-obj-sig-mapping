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
from tqdm import trange, tqdm

sys.path.append(str(Path("/home/zihangw/EvoComm/code")))
from simulations import SimulationGraphRecord
from model import Config, LanguageModelNorm

# from simulations import Logger

BASE_PATH = Path("/home/zihangw/EvoComm")
plt.rcParams.update({'font.size': 20})

# %% ----- ----- ----- ----- ----- ----- run traj ----- ----- ----- ----- ----- ----- %% #
run_graph_list = [
    "networks/bottleneck_pop100/bn_ndeme10_edge5_beta0_rep0.txt"
]
num_trials = 10

num_objects = 5
num_sounds = 5
model_name = "norm"
num_runs = 10000
sample_times = 10
temperature = 1.0
# n_trial = int(sys.argv[9])

# %%
# for graph_path in run_graph_list:
#     config = Config(num_objects, num_sounds,
#                     sample_times = sample_times, temperature = temperature)
#     for i_rep in trange(num_trials):
#         # os.system(f"python3 /home/zihangw/EvoComm/code/sim_graph.py 5 5 softmax {graph_path} 50000 /home/zihangw/EvoComm/results_evolve /home/zihangw/EvoComm/results_evolve 1 1.0 {i_rep}")

#         sim = SimulationGraphRecord(config, BASE_PATH / graph_path)
#         sim.initialize(LanguageModelNorm)
#         # fitness_list = sim.run(num_runs)
#         sim.run(num_runs)

#         graph_base = os.path.dirname(graph_path)
#         graph_name = os.path.basename(graph_path).split(".")[0]
#         file_path = BASE_PATH / "results_evolve_traj" / f"{model_name}" / f"{graph_base}" / f"{graph_name}" / f"st_{sample_times}_temp_{temperature:.1f}_{i_rep}.pkl"
#         os.makedirs(file_path.parent, exist_ok=True)

#         with open(file_path, "wb") as f:
#             # pickle.dump({"fitness_list": fitness_list, "logger": sim.logger.get_logs()}, f)
#             pickle.dump(sim.logger.get_logs(), f)

# %%
logger_list = defaultdict(list)
final_lang = defaultdict(list)
for i_graph, graph_path in enumerate(run_graph_list):
    for i_rep in trange(num_trials):
        graph_base = os.path.dirname(graph_path)
        graph_name = os.path.basename(graph_path).split(".")[0]
        file_path = BASE_PATH / "results_evolve_traj" / f"{model_name}" / f"{graph_base}" / f"{graph_name}" / f"st_{sample_times}_temp_{temperature:.1f}_{i_rep}.pkl"
        with open(file_path, "rb") as f:
            logger = pickle.load(f)
        logger_list[i_graph].append(logger)
        final_lang[i_graph].append(logger["num_languages"][-1])

# %%
i_graph = 0
rep = 0
graph_path = run_graph_list[i_graph]
logger = logger_list[i_graph][rep]
G = nx.read_edgelist(BASE_PATH / graph_path, nodetype=int)

graph_base = os.path.dirname(graph_path)
graph_base_name = graph_base.split("/")[-1]
graph_name = os.path.basename(graph_path).split(".")[0]
num_demes = int(graph_name.split("ndeme")[-1].split("_")[0])
num_edge_added = int(graph_name.split("edge")[-1].split("_")[0])
# beta = int(graph_name.split("beta")[-1].split("_")[0])
pop_size = int(graph_base_name.split("pop")[-1])
deme_size = pop_size // num_demes

num_log_data = len(logger["iteration"])
# fitness_vector = np.array(logger["fitness_vector"])
# payoff_vector = np.array(logger["payoff_vector"])
# language_tags = np.array(logger["language_tags"])
# num_languages = np.array(logger["num_languages"])
# iterations = np.array(logger["iteration"])

# def rename_numbers(lst):
#     unique_values = sorted(set(lst))  # Get unique values sorted
#     mapping = {val: i for i, val in enumerate(unique_values)}  # Create mapping
#     return [mapping[val] for val in lst]  # Replace values using mapping

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_n_demes(
    G, deme_size, n_demes,
    inter_edges=None, labels=None,
    cluster_assignments=None, cluster_cmap="tab20",
    cluster_color_map=None  # NEW: global mapping for consistency
):
    """
    Plot a graph with n demes arranged in a circle.

    Parameters:
        G (networkx.Graph): The full combined graph (all nodes + edges).
        deme_size (int): Number of nodes per deme.
        n_demes (int): Number of demes.
        inter_edges (list of tuple, optional): Edges between demes (not within).
        labels (dict, optional): Labels for nodes.
        cluster_assignments (list or dict, optional): Cluster IDs for nodes.
            - If list: index = node ID, value = cluster ID
            - If dict: {node: cluster ID}
        cluster_cmap (str, optional): Colormap for clusters (default: "tab20").
        cluster_color_map (dict, optional): Predefined mapping {cluster_id: color}.
            Useful for ensuring consistent colors across multiple plots.

    Returns:
        dict: The cluster → color mapping used in this plot.
    """
    pos = {}
    node_colors = {}
    node_border_colors = {}

    # Place deme centers on a big circle
    radius = 5
    for i in range(n_demes):
        angle = 2 * np.pi * i / n_demes
        cx, cy = radius * np.cos(angle), radius * np.sin(angle)

        deme_nodes = range(i * deme_size, (i + 1) * deme_size)
        deme_pos = nx.circular_layout(deme_nodes)

        for node in deme_pos:
            deme_pos[node][0] += cx
            deme_pos[node][1] += cy
            pos[node] = deme_pos[node]
            node_border_colors[node] = "black"

    # Assign colors
    if cluster_assignments is not None:
        # Extract cluster IDs
        if isinstance(cluster_assignments, dict):
            clusters = [cluster_assignments[node] for node in G.nodes]
        else:
            clusters = cluster_assignments

        # Initialize or extend mapping
        if cluster_color_map is None:
            cluster_color_map = {}

        # unique_clusters = sorted(set(clusters))
        # missing_clusters = [c for c in unique_clusters if c not in cluster_color_map]

        # if missing_clusters:
        #     cmap = plt.cm.get_cmap(cluster_cmap, len(cluster_color_map) + len(missing_clusters))
        #     start_idx = len(cluster_color_map)
        #     for i, c in enumerate(missing_clusters, start=start_idx):
        #         cluster_color_map[c] = cmap(i)

        cmap = plt.cm.get_cmap(cluster_cmap)  # continuous colormap

        # Assign unseen clusters a new color based on their order of appearance
        for cluster_id in clusters:
            if cluster_id not in cluster_color_map:
                idx = len(cluster_color_map)  # next unused slot
                cluster_color_map[cluster_id] = cmap(idx % cmap.N)

        # Assign node colors using mapping
        for node, cluster_id in zip(G.nodes, clusters):
            node_colors[node] = cluster_color_map[cluster_id]
    else:
        # Default: color by deme
        cmap = plt.cm.get_cmap("tab10", n_demes)
        for i in range(n_demes):
            for node in range(i * deme_size, (i + 1) * deme_size):
                node_colors[node] = cmap(i)

    # Highlight inter-deme connections
    if inter_edges is not None:
        for (n1, n2) in inter_edges:
            node_border_colors[n1] = "#8ED973"
            node_border_colors[n2] = "#8ED973"

    # Draw
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(
        G, pos,
        node_color=[node_colors[node] for node in G.nodes],
        edgecolors=[node_border_colors[node] for node in G.nodes],
        node_size=700, linewidths=2
    )
    nx.draw_networkx_edges(G, pos, edge_color="black")
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color="white")

    plt.title(f"{n_demes}-Deme Graph with Inter-Deme Connections")
    plt.axis("off")
    # plt.show()

    return cluster_color_map  # return mapping for reuse

# %%
cluster_color_map = None
for i_gen in trange(num_log_data-1,-1,-1): 
    labels = {i: np.round(fitness,2) for i, fitness in enumerate(logger["fitness_vector"][i_gen])}
    clusters = logger["language_tags"][i_gen]
    cluster_color_map = plot_n_demes(G, deme_size, num_demes, labels=labels, cluster_assignments=clusters)

    file_path = BASE_PATH / "figures" / "evolve_traj" / f"graph_{i_graph}_rep_{rep}_gen_{logger["iteration"][i_gen]}.png"
    os.makedirs(file_path.parent, exist_ok=True)
    plt.savefig(file_path, dpi=300)
    plt.close()

    # labels = {i: np.round(fitness,2) for i, fitness in enumerate(logger["fitness_vector"][-2])}
    # clusters = logger["language_tags"][-2]
    # cluster_color_map = plot_n_demes(G, deme_size, num_demes, labels=labels, cluster_assignments=clusters, cluster_color_map=cluster_color_map)  # reuse


# %% ----- ----- ----- ----- ----- ----- normal evolve results ----- ----- ----- ----- ----- ----- %% #
# param_file = BASE_PATH / "param_space" / "evolve_param_demes_fix_popsize.txt"
# out_path_base = BASE_PATH / "results_evolve"
# # combined_name = BASE_PATH / "results_invade_combined" / "invade_param_demes_multi.csv"
# # graph_info_path = BASE_PATH / "results_invade_combined" / "invade_graph_info.csv"
# pop_size_select = 100
# num_demes_select = 10
# num_edge_added_select = 5

# with open(param_file, "r") as f:
#     param_sim = f.readlines()
# param_sim = [x.strip() for x in param_sim]
# param_sim = [x.split(" ") for x in param_sim]

# data_all = defaultdict(list)
# logger_list = []
# for param in param_sim:
#     # num_objects = int(param[0])
#     # num_sounds = int(param[1])
#     model_name = param[2]
#     graph_path = param[3]
#     num_runs = int(param[4])
#     sample_times = int(param[5])
#     temperature = float(param[6])

#     graph_base = os.path.dirname(graph_path)
#     graph_base_name = graph_base.split("/")[-1]
#     graph_name = os.path.basename(graph_path).split(".")[0]
#     out_path = out_path_base / graph_base / graph_name
    
#     if "wm" in graph_name:
#         num_demes = 1
#         num_edge_added = 0
#         beta = 0
#         pop_size = int(graph_base_name.split("_")[-1])
#         deme_size = pop_size
#     else:
#         num_demes = int(graph_name.split("ndeme")[-1].split("_")[0])
#         num_edge_added = int(graph_name.split("edge")[-1].split("_")[0])
#         beta = int(graph_name.split("beta")[-1].split("_")[0])
#         pop_size = int(graph_base_name.split("pop")[-1])
#         deme_size = pop_size // num_demes
    
#     if (pop_size == pop_size_select) and (num_demes == num_demes_select) and (num_edge_added == num_edge_added_select):
#         for i in range(1000):
#             graph_base = os.path.dirname(graph_path)
#             graph_name = os.path.basename(graph_path).split(".")[0]

#             out_path = os.path.join(out_path_base, model_name, graph_base, graph_name)
#             file_path = os.path.join(out_path, f"st_{sample_times}_temp_{temperature:.1f}_{i}.pkl")

#             try:
#                 with open(file_path, "rb") as f:
#                     logger = pickle.load(f)
#                 logger_list.append(logger)
#             except FileNotFoundError:
#                 continue
#         break

# final_lang = []
# for logger in logger_list:
#     final_lang.append(logger["num_languages"][-1])

# %%
