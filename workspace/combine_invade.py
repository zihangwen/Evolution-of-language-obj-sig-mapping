# %%
import numpy as np
import os
import sys
import seaborn as sns
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import re


base_path = "/home/zihangw/EvoComm"
# %%
param_file = "param_space/param_bottleneck_demes_invade.txt"  # Path to your parameter file
out_path_base = "results_invade"
n_trial = 100

figure_path_base = f"/home/zihangw/EvoComm/figures_invading/"

with open(os.path.join(base_path, param_file), "r") as f:
    param_lines = [line.strip() for line in f.readlines()]

# %%
data = []
for line in param_lines:
    params = line.split(" ")
    num_objects = int(params[0])
    num_sounds = int(params[1])
    graph_path = params[2]
    num_trials = int(params[3])

    graph_base = os.path.dirname(graph_path)
    graph_name = os.path.basename(graph_path).split(".")[0]
    out_path = os.path.join(base_path, out_path_base, graph_base, graph_name)

    # graph property
    G = nx.read_edgelist(os.path.join(base_path, graph_path), nodetype=int)
    mean_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()

    # Compute the Laplacian matrix
    L = nx.laplacian_matrix(G).toarray()

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(L)

    # Algebraic connectivity is the second smallest eigenvalue
    lambda_2 = sorted(eigenvalues)[1]

    # Read results
    total_count = 0
    weighted_sum_time = 0  # To compute weighted average
    num_trials_done = 0
    for i_trial in range(n_trial):
        try:
            with open(os.path.join(out_path, f"{i_trial}.txt"), "r") as fr:
                result_lines = [line.strip() for line in fr.readlines()]
        except FileNotFoundError:
            continue
        
        try:
            assert len(result_lines) >= 2
        except AssertionError:
            print(f"Error in {out_path}/{i_trial}.txt")
            continue

        _, _, count, avg_time = result_lines[1].split("\t")
        count = int(count)
        avg_time = float(avg_time)
        num_trials_done += num_trials

        total_count += count
        weighted_sum_time += count * avg_time
    
    overall_avg_time = weighted_sum_time / total_count if total_count > 0 else 0
    pfix = total_count / num_trials_done

    data.append([num_objects, num_sounds, graph_base.split("/")[-1], graph_name, mean_degree, lambda_2, num_trials_done, total_count, pfix, overall_avg_time])

df = pd.DataFrame(data, columns=["num_objects", "num_sounds", "graph_type", "graph_name", "graph_mean_degree", "Algebraic connectivity", "num_trials_done", "total_count", "pfix", "time to fix"])

# %%
df = df.sort_values(by="graph_name", ascending=True)  # Sort by 'Total_Count'
df = df.reset_index(drop=True)  # Reset index

# %%
df_bottleneck = df[df["graph_type"] == "bottleneck_demes"].copy()
graph_names = df_bottleneck["graph_name"].tolist()
graph_number_list = [re.findall(r"-?\d+\.?\d*", i_gn) for i_gn in graph_names]
n_demes, edge_added, beta, rep = zip(*graph_number_list)
df_bottleneck["n_demes"] = [int(x) for x in n_demes]
df_bottleneck["edge_added"] = [int(x) for x in edge_added]
df_bottleneck["beta"] = [float(x) for x in beta]
df_bottleneck["rep"] = [int(x) for x in rep]


# %% bottleneck graphs
sns.set_theme(style="whitegrid")
df_filtered = df_bottleneck[df_bottleneck["beta"] == 0]
x_options = ['n_demes', 'Algebraic connectivity']
y_options = ['pfix', 'time to fix']
figure_name = "bottleneck_demes"
# figure_path = os.path.join(figure_path_base, figure_name)
os.makedirs(figure_path_base, exist_ok=True)

fig, axes = plt.subplots(len(y_options), len(x_options), figsize=(12, 8), sharex='col')

# legend_added = False
for i, x in enumerate(x_options):
    for j, y in enumerate(y_options):
        ax = axes[j, i]
        sns.lineplot(data=df_filtered, x=x, y=y, ax=ax, marker='o', errorbar="se")
        ax.set_title(f"{y} vs {x}")
        ax.set_xlabel(x)
        ax.set_ylabel(y)

        # if not legend_added:
        #     legend_added = True
        # else:
        #     ax.legend_.remove()  # Remove legend from all other plots

fig.tight_layout()
plt.show()
fig.savefig(os.path.join(figure_path_base, f"{figure_name}_alge_conn.jpg"), dpi=600)
# plt.figure(figsize=(12, 6))
# sns.lineplot(x="graph_mean_degree", y="pfix", data=df_filtered)

# %%
