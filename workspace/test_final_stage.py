# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import re

# %%
num_objects = 5
num_sounds = 5

network_name = "bottleneck"
model_name_list = ["norm", "softmax"]
# graph_path_list = ["networks/toy/" + a for a in os.listdir("/home/zihangw/EvoComm/networks/toy")]

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

# %%
number_list = [tuple(map(int, re.findall(r"-?\d+", s))) for s in graph_name_list]
num_edge_added_list = [number[0] for number in number_list]
beta_list = [number[1] for number in number_list]

# %%
model_dict = {}
df_all = pd.DataFrame()
for model_name in model_name_list:
    # log_dict = {}
    for i_graph in graph_name_list:
        # log_dict[i_graph] = []
        num_edge_added, beta, graph_rep = tuple(map(int, re.findall(r"-?\d+", i_graph)))
        log_list = []
        for n_trial in range(n_trials):
            for i_trial in range(num_trials):
                # graph_name = os.path.basename(graph_path).split(".")[0]
                out_path = os.path.join(out_path_base, model_name, i_graph)
                log_temp = np.loadtxt(os.path.join(out_path, "%d.txt" %(n_trial*num_trials+i_trial)))[:500]
                # log_dict[i_graph].append(log_temp)
                log_list.append(log_temp[-1])
        
        log_dict_mean = np.mean(log_list, 0)
        log_dict_std = np.std(log_list, 0)
        new_row = pd.DataFrame({
            "deme_number": [2],
            "num_edge_added": [num_edge_added],
            "beta": [beta],
            "graph_rep": [graph_rep],
            "model": [model_name],
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