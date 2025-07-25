# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import re
import seaborn as sns

# %%
num_objects = 5
num_sounds = 5

network_name = "bottleneck_demes"
model_name_list = ["norm", "softmax"]
# graph_path_list = ["networks/toy/" + a for a in os.listdir("/home/zihangw/EvoComm/networks/toy")]
temperature_list = [1.0]
# temperature_list = [1.0, 2.0, 5.0, 10.0]
sample_times_list = [100]
# sample_times_list = [1, 5, 10, 20, 40, 100]

graph_name_list = [a for a in os.listdir(f"/home/zihangw/EvoComm/results/{network_name}/norm")]
# graph_name_list = sorted(graph_name_list, key=lambda s: tuple(map(int, re.findall(r"\d+", s))))
graph_name_list = sorted(graph_name_list, key=lambda s: tuple(map(int, re.findall(r"-?\d+", s))))
# graph_path_list = ["graphs/3-r.txt", "graphs/5-r.txt", "graphs/detour.txt", "graphs/star.txt", "graphs/wheel.txt", "graphs/wm_100.txt"]
# graph_path_list = ["graphs/detour.txt", "graphs/wm_100.txt"]
# num_runs = int(1e5)
out_path_base = f"/home/zihangw/EvoComm/results/{network_name}"

figure_path = f"/home/zihangw/EvoComm/figures_final_stage/{network_name}"
os.makedirs(figure_path, exist_ok=True)
n_trials = 10
num_trials = 10
trials = n_trials * num_trials

# %%
# number_list = [tuple(map(int, re.findall(r"-?\d+", s))) for s in graph_name_list]
# num_edge_added_list = [number[0] for number in number_list]
# beta_list = [number[1] for number in number_list]

# %%
model_dict = {}
df_all = pd.DataFrame()
for model_name in model_name_list:
    # log_dict = {}
    for i_graph in graph_name_list:
        # log_dict[i_graph] = []
        deme_number, num_edge_added, beta, graph_rep = tuple(map(int, re.findall(r"-?\d+", i_graph)))
        for sample_times in sample_times_list:
            for temperature in temperature_list:
                st_name = f"st_{sample_times}_temp_{temperature:.1f}"
                # gst_name = f"{i_graph}_{st_name}"
                log_list = []
                for trial in range(trials):
                    # graph_name = os.path.basename(graph_path).split(".")[0]
                    out_path = os.path.join(out_path_base, model_name, i_graph)
                    log_temp = np.loadtxt(os.path.join(out_path, f"{st_name}_{trial}.txt"))[:500]
                    # log_dict[i_graph].append(log_temp)
                    log_list.append(log_temp[-1])
        
                log_dict_mean = np.mean(log_list, 0)
                log_dict_std = np.std(log_list, 0)
                new_row = pd.DataFrame({
                    "deme_number": [deme_number],
                    "num_edge_added": [num_edge_added],
                    "beta": [beta],
                    "graph_rep": [graph_rep],
                    "model": [model_name],
                    "sample_times": [sample_times],
                    "temperature": [temperature],
                    "mean_fitness_mean": [log_dict_mean[1]],
                    "mean_fitness_std": [log_dict_std[1]],
                    "max_fitness_mean": [log_dict_mean[2]],
                    "max_fitness_std": [log_dict_std[2]],
                    "num_langauge_mean": [log_dict_mean[3]],
                    "num_langauge_std": [log_dict_std[3]],
                })
                df_all = pd.concat([df_all, new_row], ignore_index=True)
    
    # df = pd.DataFrame(columns=["deme_number", "num_edge_added", "beta", "graph_rep"])
    # df["num_edge_added"] = num_edge_added_list
    # df["beta"] = beta_list
    # df["deme_number"] = 2
    # df["graph_rep"] = 0
    # df["model"] = model_name

    # model_dict[model_name] = log_dict

# %%
# Set style
# sns.set_theme(style="whitegrid")

# # Define x-axis options
# x_options = ['num_edge_added', 'beta']
# y_options = ['mean_fitness_mean', 'max_fitness_mean', 'num_langauge_mean']

# df_filtered = df_all[df_all['deme_number'] == 10]

# # Create subplots
# fig, axes = plt.subplots(len(y_options), len(x_options), figsize=(12, 8), sharex='col')

# legend_added = False
# for i, x in enumerate(x_options):
#     for j, y in enumerate(y_options):
#         ax = axes[j, i]
#         sns.lineplot(data=df_filtered, x=x, y=y, ax=ax, hue="model", marker='o')
#         ax.set_title(f"{y} vs {x}")
#         ax.set_xlabel(x)
#         ax.set_ylabel(y)

#         if not legend_added:
#             legend_added = True
#         else:
#             ax.legend_.remove()  # Remove legend from all other plots

# fig.tight_layout()
# plt.show()
# fig.savefig(os.path.join(figure_path, f"final_stage_{network_name}.jpg"), dpi=600)

# %%
# Set theme
sns.set_theme(style="whitegrid")

# Filter for beta = 0
df_filtered = df_all[df_all["beta"] == 0]

# Get unique deme_numbers
deme_numbers = sorted(df_filtered["deme_number"].unique())

# Define y-axis options
# y_options = ['mean_fitness_mean', 'max_fitness_mean', 'num_langauge_mean']
y_options = [
    ("mean_fitness_mean", "mean_fitness_std"),
    ("max_fitness_mean", "max_fitness_std"),
    ("num_langauge_mean", "num_langauge_std")
]


# Create subplots (rows = y-options, columns = deme_number)
fig, axes = plt.subplots(len(y_options), len(deme_numbers), figsize=(12, 8), sharex='col')

legend_added = False
for i, deme in enumerate(deme_numbers):
    df_deme = df_filtered[df_filtered["deme_number"] == deme]  # Filter by deme_number
    
    for j, (y, y_err) in enumerate(y_options):
        ax = axes[j, i]
        sns.lineplot(data=df_deme, x="num_edge_added", y=y, hue="model", ax=ax, marker='o', errorbar=("sd"))
        ax.set_title(f"{y} ({deme} {int(100 / deme)}-Demes)")
        ax.set_xlabel("num_edge_added")
        ax.set_ylabel(y)

        # Add shaded error regions manually
        for model in df_deme["model"].unique():
            df_model = df_deme[df_deme["model"] == model]
            ax.fill_between(
                df_model["num_edge_added"], 
                df_model[y] - df_model[y_err], 
                df_model[y] + df_model[y_err], 
                alpha=0.2  # Transparency
            )


        if not legend_added:
            legend_added = True
        else:
            ax.legend_.remove()  # Remove legend from all other plots

# Adjust layout
plt.tight_layout()
plt.show()
fig.savefig(os.path.join(figure_path, f"final_stage.jpg"), dpi=600)