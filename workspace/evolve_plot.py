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
    max_fitness = []
    max_self_payoff = []
    counter = 0
    for i in range(2000):
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

    data_all["graph_name"].append(graph_base_name)
    data_all["num_trials"].append(counter)
    data_all["num_demes"].append(num_demes)
    data_all["deme_size"].append(deme_size)
    data_all["num_edge_added"].append(num_edge_added)
    data_all["beta"].append(beta)
    data_all["num_generations"].append(num_runs)

    data_all["final_num_lang"].append(np.mean(final_lang).item())
    data_all["final_num_lang_time"].append(np.mean(final_lang_time).item())
    data_all["max_fitness"].append(np.mean(max_fitness).item())
    data_all["max_self_payoff"].append(np.mean(max_self_payoff).item())

# %%
df = pd.DataFrame(data_all)

# %%
base_line_df = df[df["num_demes"] == 1].copy()
base_line_df = base_line_df.reset_index(drop=True)
# df = pd.concat([df, base_line_df, base_line_df]).drop_duplicates(keep=False)
df = df[df["num_demes"] > 1]
df = df.reset_index(drop=True)

# %%
name_mapping_dict = {
    "num_demes": "Number of Demes",
    "deme_size": "Deme Size",
    "num_edge_added": "Number of Edges Added Between Each Two Demes",
    "final_num_lang": "Final Number of Languages",
    "final_num_lang_time": "Time to Reach Final Number of Languages",
    "max_fitness": "Max Fitness",
    "max_self_payoff": "Max Self Payoff",
}
# %%
legend_item = "num_demes"
panel_item = 'deme_size'
x_item = "num_edge_added"
# y_item = "final_num_lang"

# panel_item = "deme_size"
# panel_item = "num_demes"
# x_item = "num_edge_added"
panel_item_list = df[panel_item].unique()[::-1]
base_panel_item_list = base_line_df[panel_item].unique()[::-1]

y_item_list = ["final_num_lang", "final_num_lang_time", "max_fitness", "max_self_payoff"]

for y_item in y_item_list:

    fig, axes = plt.subplots(1, len(panel_item_list), figsize=(12 * len(panel_item_list), 8), sharex='col')
    # plt.figure(figsize=(12, 8))
    for i, i_panel in enumerate(panel_item_list):
        ax = axes[i]
        df_select = df[df[panel_item] == i_panel]
        sns.lineplot(data=df_select, x=x_item, y=y_item, hue=legend_item, markers=True, dashes=False, ax=ax, palette="Set1")
        ax.set_title(f"{name_mapping_dict[panel_item]}: {i_panel}")
        ax.set_xlabel(f"{name_mapping_dict[x_item]}")
        ax.set_ylabel(name_mapping_dict[y_item])

    for i, i_panel in enumerate(base_panel_item_list):
        ax = axes[i]
        base_df_select = base_line_df[base_line_df[panel_item] == i_panel]
        ax.axhline(y=base_df_select[y_item].item(), color='black', linestyle='--', label='Baseline')
        ax.get_legend().remove()

    ax.legend(title=name_mapping_dict[legend_item], bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    plt.savefig(BASE_PATH / "figures" / f"evolve_{y_item}_vs_{x_item}_{panel_item}_{legend_item}.png", dpi = 300, bbox_inches='tight')
    plt.show()


# %% fix num_edge_added
num_edge_added_select = 10
df_select = df[df["num_edge_added"] == num_edge_added_select]

x_item = "deme_size"
legend_item = "num_demes"
# y_item = "final_num_lang"

y_item_list = ["final_num_lang", "final_num_lang_time", "max_fitness", "max_self_payoff"]

fig, axes = plt.subplots(1, len(y_item_list), figsize=(12 * len(panel_item_list), 8), sharex='col')
for i, y_item in enumerate(y_item_list):
    ax = axes[i]
    
    sns.lineplot(data=df_select, x=x_item, y=y_item, hue=legend_item, markers=True, dashes=False, ax=ax, palette="Set1")
    ax.set_title(name_mapping_dict[y_item])
    ax.set_xlabel(f"{name_mapping_dict[x_item]}")
    ax.set_ylabel(name_mapping_dict[y_item])

    ax.plot(base_df_select[x_item], base_df_select[y_item], color='black', linestyle='--', label='Baseline')
    ax.get_legend().remove()

ax.legend(title=name_mapping_dict[legend_item], bbox_to_anchor=(1.05, 1), loc='upper left')
fig.suptitle(f"{name_mapping_dict["num_edge_added"]}: {num_edge_added_select}")
fig.tight_layout()
plt.savefig(BASE_PATH / "figures" / f"evolve_x_{x_item}_legend_{legend_item}.png", dpi = 300, bbox_inches='tight')
plt.show()

# %%
