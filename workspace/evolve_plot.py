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
plt.rcParams.update({'font.size': 20})

# %%
param_file = BASE_PATH / "param_space" / "evolve_param_demes_fix_popsize.txt"
out_path_base = BASE_PATH / "results_evolve"
# combined_name = BASE_PATH / "results_invade_combined" / "invade_param_demes_multi.csv"
# graph_info_path = BASE_PATH / "results_invade_combined" / "invade_graph_info.csv"
pop_size_select = 100

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
    
    if "wm" in graph_name:
        num_demes = 1
        num_edge_added = 0
        beta = 0
        pop_size = int(graph_base_name.split("_")[-1])
        deme_size = pop_size
    else:
        num_demes = int(graph_name.split("ndeme")[-1].split("_")[0])
        num_edge_added = int(graph_name.split("edge")[-1].split("_")[0])
        beta = int(graph_name.split("beta")[-1].split("_")[0])
        pop_size = int(graph_base_name.split("pop")[-1])
        deme_size = pop_size // num_demes
    
    if pop_size != pop_size_select:
        continue

    final_lang = []
    final_lang_time = []
    max_fitness = []
    max_self_payoff = []
    counter = 0
    for i in range(1000):
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
        final_lang.append(logger["num_languages"][-1])
        final_lang_time.append(logger["iteration"][np.where(np.array(logger["num_languages"]) == logger["num_languages"][-1])[0][0]])
        max_fitness.append(logger["max_fitness"][-1])
        max_self_payoff.append(logger["max_self_payoff"][-1])

        # final_lang_time = logger["iteration"][np.where(np.array(logger["num_languages"]) == logger["num_languages"][-1])[0][0]]
        # max_fitness = logger["max_fitness"][-1].item()
        # max_self_payoff = logger["max_self_payoff"][-1].item()

    G = nx.read_edgelist(BASE_PATH / graph_path, nodetype=int)
    mean_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()

    L = nx.laplacian_matrix(G).toarray()

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(L)

    # Algebraic connectivity is the second smallest eigenvalue
    lambda_2 = sorted(eigenvalues)[1]

    # transitivity
    transitivity = nx.transitivity(G)

    data_all["graph_base"].append(graph_base_name.split("_")[0])
    data_all["num_trials"].append(counter)
    data_all["pop_size"].append(pop_size)
    data_all["num_demes"].append(num_demes)
    data_all["deme_size"].append(deme_size)
    data_all["num_edge_added"].append(num_edge_added)
    data_all["beta"].append(beta)
    data_all["graph_mean_degree"].append(mean_degree)
    data_all["lambda_2"].append(lambda_2)
    data_all["transitivity"].append(transitivity)

    data_all["num_generations"].append(num_runs)
    data_all["final_num_lang"].append(np.mean(final_lang).item())
    data_all["final_num_lang_time"].append(np.mean(final_lang_time).item())
    data_all["max_fitness"].append(np.mean(max_fitness).item())
    data_all["max_self_payoff"].append(np.mean(max_self_payoff).item())

# %%
df = pd.DataFrame(data_all)
df_base_line = df[df["graph_base"] == "wm"].reset_index(drop=True)
df = df[df["graph_base"] == "bottleneck"].reset_index(drop=True)

# %%
# df_base_line = df[df["num_demes"] == 1].copy()
# df_base_line = df_base_line.reset_index(drop=True)
# df = pd.concat([df, df_base_line, df_base_line]).drop_duplicates(keep=False)
# df = df[df["num_demes"] > 1]
# df = df.reset_index(drop=True)

# %%
name_mapping_dict = {
    "num_demes": "Number of Demes",
    "deme_size": "Deme Size",
    "pop_size": "Total Population Size",
    "num_edge_added": "Number of Edges Added Between Each Two Demes",
    "beta": "Beta",
    "graph_mean_degree": "Graph Mean Degree",
    "lambda_2": "Algebraic Connectivity",
    "transitivity": "Transitivity",
    "final_num_lang": "Final Number of Languages",
    "final_num_lang_time": "Time to Reach Final Number of Languages",
    "max_fitness": "Max Fitness",
    "max_self_payoff": "Max Self Payoff",
}

# # %%
# x_item = "num_edge_added"
# legend_item = "num_demes"
# panel_item = "pop_size"

# # panel_item = "deme_size"
# # panel_item = "num_demes"
# # x_item = "num_edge_added"
# panel_item_list = df[panel_item].unique()[::-1]
# base_panel_item_list = df_base_line[panel_item].unique()[::-1]

# y_item_list = ["final_num_lang", "final_num_lang_time", "max_fitness", "max_self_payoff"]

# for y_item in y_item_list:

#     fig, axes = plt.subplots(1, len(panel_item_list), figsize=(12 * len(panel_item_list), 8), sharex='col')
#     # plt.figure(figsize=(12, 8))
#     for i, i_panel in enumerate(panel_item_list):
#         ax = axes[i]
#         df_select = df[df[panel_item] == i_panel]
#         sns.lineplot(data=df_select, x=x_item, y=y_item, hue=legend_item, markers=True, dashes=False, ax=ax, palette="Set1")
#         ax.set_title(f"{name_mapping_dict[panel_item]}: {i_panel}")
#         ax.set_xlabel(f"{name_mapping_dict[x_item]}")
#         ax.set_ylabel(name_mapping_dict[y_item])

#     for i, i_panel in enumerate(base_panel_item_list):
#         ax = axes[i]
#         df_base_line = df_base_line[df_base_line[panel_item] == i_panel]
#         ax.axhline(y=df_base_line[y_item].item(), color='black', linestyle='--', label='Baseline')
#         # ax.get_legend().remove()
#         ax.legend(title=name_mapping_dict[legend_item], loc="best")

#     # ax.legend(title=name_mapping_dict[legend_item], bbox_to_anchor=(1.05, 1), loc='upper left')
#     fig.tight_layout()
#     # plt.savefig(BASE_PATH / "figures" / f"evolve_{y_item}_vs_{x_item}_{panel_item}_{legend_item}.png", dpi = 300, bbox_inches='tight')
#     plt.show()


# %% fix num_edge_added
# num_edge_added_select = 10
# df_select = df[df["num_edge_added"] == num_edge_added_select]

# x_item = "num_demes"
# legend_item = "pop_size"
# # y_item = "final_num_lang"

# y_item_list = ["final_num_lang", "final_num_lang_time", "max_fitness", "max_self_payoff"]

# fig, axes = plt.subplots(1, len(y_item_list), figsize=(12 * len(y_item_list), 8), sharex='col')
# for i, y_item in enumerate(y_item_list):
#     ax = axes[i]
    
#     sns.lineplot(data=df_select, x=x_item, y=y_item, hue=legend_item, markers=True, dashes=False, ax=ax, palette="Set1")
#     ax.set_title(name_mapping_dict[y_item])
#     ax.set_xlabel(f"{name_mapping_dict[x_item]}")
#     ax.set_ylabel(name_mapping_dict[y_item])

#     ax.plot(df_base_line[x_item], df_base_line[y_item], color='black', linestyle='--', label='Baseline')
#     # ax.get_legend().remove()

# # ax.legend(title=name_mapping_dict[legend_item], bbox_to_anchor=(1.05, 1), loc='upper left')
# fig.suptitle(f"{name_mapping_dict["num_edge_added"]}: {num_edge_added_select}")
# fig.tight_layout()
# plt.savefig(BASE_PATH / "figures" / f"evolve_x_{x_item}_legend_{legend_item}.png", dpi = 300, bbox_inches='tight')
# plt.show()

# %%
pop_size_select = 100
df_select = df[df["pop_size"] == pop_size_select]
df_base_line_select = df_base_line[df_base_line["pop_size"] == pop_size_select]

# legend_item = "num_demes"
# x_item = "num_edge_added"

# legend_item = "num_edge_added"
# x_item = "num_demes"

x_item = "transitivity"
legend_item = "num_demes"

y_item_list = ["final_num_lang", "final_num_lang_time", "max_fitness", "max_self_payoff"]

fig, axes = plt.subplots(1, len(y_item_list), figsize=(12 * len(y_item_list), 8), sharex='col')
for i, y_item in enumerate(y_item_list):
    ax = axes[i]
    
    sns.lineplot(data=df_select, x=x_item, y=y_item, hue=legend_item, markers=True, dashes=False, ax=ax, palette="Set1")
    ax.set_title(name_mapping_dict[y_item])
    ax.set_xlabel(f"{name_mapping_dict[x_item]}")
    ax.set_ylabel(name_mapping_dict[y_item])

    ax.axhline(y=df_base_line_select[y_item].item(), color='black', linestyle='--', label='Baseline')
    # ax.plot(df_base_line[x_item], df_base_line[y_item], color='black', linestyle='--', label='Baseline')
    ax.get_legend().remove()

ax.legend(title=name_mapping_dict[legend_item], bbox_to_anchor=(1.05, 1), loc='upper left')
# fig.suptitle(f"{name_mapping_dict["num_edge_added"]}: {num_edge_added_select}")
fig.tight_layout()
plt.savefig(BASE_PATH / "figures" / f"evolve_x_{x_item}_legend_{legend_item}_pop_{pop_size_select}.png", dpi = 300, bbox_inches='tight')
plt.show()

# %%