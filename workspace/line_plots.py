# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import re

# %%
def filter_by_second_number(strings, pos = 1, target=1):
    return [s for s in strings if len(re.findall(r"-?\d+", s)) > 1 and int(re.findall(r"-?\d+", s)[pos]) == target]

# %%
num_objects = 5
num_sounds = 5

network_name = "temp_sample_size"
model_name_list = ["norm", "softmax"]

temperature_list = [1.0]
# temperature_list = [1.0, 2.0, 5.0, 10.0]
# sample_times_list = [100]
sample_times_list = [1, 5, 10, 20, 40, 100]

graph_name_list = ["reg_tri_100_4_0"]


# graph_name_list = [a for a in os.listdir(f"/home/zihangw/EvoComm/results/{network_name}/norm")]
# graph_name_list = filter_by_second_number(graph_name_list, pos = 2, target=0) # beta = 0
# graph_name_list = sorted(graph_name_list, key=lambda s: tuple(map(int, re.findall(r"-?\d+", s))))

# graph_path_list = ["networks/toy/" + a for a in os.listdir("/home/zihangw/EvoComm/networks/toy")]
# graph_name_list = sorted(graph_name_list, key=lambda s: tuple(map(int, re.findall(r"\d+", s))))

# num_runs = int(1e5)
out_path_base = f"/home/zihangw/EvoComm/results/{network_name}"

figure_path = f"/home/zihangw/EvoComm/figures/{network_name}"
os.makedirs(figure_path, exist_ok=True)
n_trials = 10
num_trials = 10
trials = n_trials * num_trials
# %%
# graph_name_list = [os.path.basename(graph_path).split(".")[0] for graph_path in graph_path_list]
# graph_name_list = sorted(graph_name_list, key=lambda s: tuple(map(int, re.findall(r"\d+", s)[:3])))

# %%
model_dict = {}
for model_name in model_name_list:
    log_dict = {}
    for i_graph in graph_name_list:
        for sample_times in sample_times_list:
            for temperature in temperature_list:
                st_name = f"st_{sample_times}_temp_{temperature:.1f}"
                # gst_name = f"{i_graph}_{st_name}"
                # gst_name = f"{i_graph}"
                gst_name = f"{i_graph}_{sample_times}"
                log_dict[gst_name] = []
                for trial in range(trials):
                    # graph_name = os.path.basename(graph_path).split(".")[0]
                    out_path = os.path.join(out_path_base, model_name, i_graph)
                    log_temp = np.loadtxt(os.path.join(out_path, f"{st_name}_{trial}.txt"))
                    log_dict[gst_name].append(log_temp)

    model_dict[model_name] = log_dict
    # print("hello world!")

# %%
wm_100_dict = {}
for model_name in model_name_list:
    wm_100_dict[model_name] = []
    for n_trial in range(10):
        for i_trial in range(10):            
            # graph_name = os.path.basename(graph_path).split(".")[0]
            out_path = os.path.join("/home/zihangw/EvoComm/results_graph/", model_name, "wm_100")
            log_temp = np.loadtxt(os.path.join(out_path, "%d.txt" %(n_trial*num_trials+i_trial)))[:500]
            wm_100_dict[model_name].append(log_temp)

# %%
star_100_dict = {}
for model_name in model_name_list:
    star_100_dict[model_name] = []
    for n_trial in range(10):
        for i_trial in range(10):            
            # graph_name = os.path.basename(graph_path).split(".")[0]
            out_path = os.path.join("/home/zihangw/EvoComm/results_graph/", model_name, "star")
            log_temp = np.loadtxt(os.path.join(out_path, "%d.txt" %(n_trial*num_trials+i_trial)))[:500]
            star_100_dict[model_name].append(log_temp)

# %%
# wm_10_dict = {}
# for model_name in model_name_list:
#     wm_10_dict[model_name] = []
#     for n_trial in range(100):
#         for i_trial in range(100):            
#             # graph_name = os.path.basename(graph_path).split(".")[0]
#             out_path = os.path.join("/home/zihangw/EvoComm/results/toy", model_name, "wm_10")
#             log_temp = np.loadtxt(os.path.join(out_path, "%d.txt" %(n_trial*num_trials+i_trial)))
#             wm_10_dict[model_name].append(log_temp)


# %%
# star_10_dict = {}
# for model_name in model_name_list:
#     star_10_dict[model_name] = []
#     for n_trial in range(100):
#         for i_trial in range(100):            
#             # graph_name = os.path.basename(graph_path).split(".")[0]
#             out_path = os.path.join("/home/zihangw/EvoComm/results/toy", model_name, "star_10")
#             log_temp = np.loadtxt(os.path.join(out_path, "%d.txt" %(n_trial*num_trials+i_trial)))
#             star_10_dict[model_name].append(log_temp)

# %%
wm_dict = wm_100_dict
star_dict = star_100_dict

# %%
# num_lines = len(graph_name_list)
num_lines = len(model_dict["softmax"])
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
colors = cm.viridis(np.linspace(0, 1, num_lines))  # Change 'viridis' to other colormaps if needed


# %%
for model_name in model_name_list:
    # model_name = "softmax"
    # if model_name == "norm":
    #     max_x = 25_000 // 100
    # else:
    #     max_x = 2000 // 100
        
    log_dict = model_dict[model_name]

    fig, ax = plt.subplots(figsize=(8, 6))
    for ii, i_graph in enumerate(log_dict.keys()):
        log_list_temp = log_dict[i_graph]
        data_ave = np.mean(log_list_temp, axis=0)
        ax.plot(data_ave[:,0], data_ave[:,2], color=colors[ii], label = i_graph)
        # ax.plot(data_ave[:,0], data_ave[:,2], label = i_graph)

    # ----- ----- #
    wm_ave = np.mean(wm_dict[model_name], axis=0)
    ax.plot(wm_ave[:,0], wm_ave[:,2], color="black", label = "wm_10")
    star_ave = np.mean(star_dict[model_name], axis=0)
    ax.plot(star_ave[:,0], star_ave[:,2], color="red", label = "star_10")
    # ----- ----- #

    # ax.axhline(-0.6, linestyle="--", color="black", label="-0.6")
    # ax.axvline(2000, linestyle="--", color="black")
    ax.set_title("model: %s" %model_name)
    ax.set_xlabel("generation")
    ax.set_ylabel("mean fitness across population")
    ax.set_ylim(0.8,4.8)
    ax.legend(loc = "upper left", bbox_to_anchor=(1, 1), fontsize = 8)
    # fig.show()
    fig.tight_layout()
    fig.savefig(os.path.join(figure_path, "fitness_%s_%s.jpg"%(model_name, str("mean"))), dpi = 600)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    for ii, i_graph in enumerate(log_dict.keys()):
        log_list_temp = log_dict[i_graph]
        data_ave = np.mean(log_list_temp, axis=0)
        ax.plot(data_ave[:,0], data_ave[:,1], color=colors[ii], label = i_graph)
        # ax.plot(data_ave[:,0], data_ave[:,1], label = i_graph)

    # ----- ----- #
    wm_ave = np.mean(wm_dict[model_name], axis=0)
    ax.plot(wm_ave[:,0], wm_ave[:,1], color="black", label = "wm_10")
    star_ave = np.mean(star_dict[model_name], axis=0)
    ax.plot(star_ave[:,0], star_ave[:,1], color="red", label = "star_10")
    # ----- ----- #

    ax.set_title("model: %s" %model_name)
    ax.set_xlabel("generation")
    ax.set_ylabel("max fitness across population")
    ax.set_ylim(0.8,4.8)
    ax.legend(loc = "upper left", bbox_to_anchor=(1, 1), fontsize = 8)
    # fig.show()
    fig.tight_layout()
    fig.savefig(os.path.join(figure_path, "fitness_%s_%s.jpg"%(model_name, str("max"))), dpi = 600)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    for ii, i_graph in enumerate(log_dict.keys()):
        log_list_temp = np.array(log_dict[i_graph])
        num_langs_traj = log_list_temp[:,:,[0,-1]]
        # non_nan_idx = ~np.isnan(num_langs_traj[0,:,-1])
        # num_langs_traj = num_langs_traj[:,non_nan_idx,:]
        
        num_langs_traj = np.mean(num_langs_traj, axis=0)
        ax.plot(num_langs_traj[:,0], num_langs_traj[:,-1], color=colors[ii], label = i_graph)
        # ax.plot(num_langs_traj[:,0], num_langs_traj[:,-1], label = i_graph)

    # ----- ----- #
    wm_ave = np.mean(wm_dict[model_name], axis=0)
    ax.plot(wm_ave[:,0], wm_ave[:,-1], color="black", label = "wm_10")
    star_ave = np.mean(star_dict[model_name], axis=0)
    ax.plot(star_ave[:,0], star_ave[:,-1], color="red", label = "star_10")
    # ----- ----- #

    ax.set_title("model: %s" %model_name)
    # ax.axhline(-0.6, linestyle="--", color="black", label="-0.6")
    # ax.axvline(2000, linestyle="--", color="black")
    ax.set_xlabel("generation")
    ax.set_ylabel("number of languages")
    # ax.set_xlim(0, 20000)
    ax.legend(loc = "upper left", bbox_to_anchor=(1, 1), fontsize = 8)
    # fig.show()
    fig.tight_layout()
    fig.savefig(os.path.join(figure_path, "num_langs_%s_%s.jpg"%(model_name, str("mean"))), dpi = 600)
    plt.close()

# %%
for model_name in model_name_list:
    # model_name = "softmax"
    if model_name == "norm":
        min_x = 0
        max_x = 50_000
        min_y = 0
        max_y = 20
    else:
        min_x = 0
        max_x = 50_000
        min_y = 0
        max_y = 10
        
    log_dict = model_dict[model_name]


    fig, ax = plt.subplots(figsize=(8, 6))
    for ii, i_graph in enumerate(log_dict.keys()):
        log_list_temp = np.array(log_dict[i_graph])
        num_langs_traj = log_list_temp[:,:,[0,-1]]
        # non_nan_idx = ~np.isnan(num_langs_traj[0,:,-1])
        # num_langs_traj = num_langs_traj[:,non_nan_idx,:]
        
        num_langs_traj = np.mean(num_langs_traj, axis=0)
        ax.plot(num_langs_traj[:,0], num_langs_traj[:,-1], color=colors[ii], label = i_graph)
        # ax.plot(num_langs_traj[:,0], num_langs_traj[:,-1], label = i_graph)

    # ----- ----- #
    wm_ave = np.mean(wm_dict[model_name], axis=0)
    ax.plot(wm_ave[:,0], wm_ave[:,-1], color="black", label = "wm_10")
    star_ave = np.mean(star_dict[model_name], axis=0)
    ax.plot(star_ave[:,0], star_ave[:,-1], color="red", label = "star_10")
    # ----- ----- #

    ax.set_title("model: %s" %model_name)
    # ax.axhline(-0.6, linestyle="--", color="black", label="-0.6")
    # ax.axvline(2000, linestyle="--", color="black")
    ax.set_xlabel("generation")
    ax.set_ylabel("number of languages")
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.legend(loc = "upper left", bbox_to_anchor=(1, 1), fontsize = 8)
    # fig.show()
    fig.tight_layout()
    fig.savefig(os.path.join(figure_path, "num_langs_%s_%s_zoomin.jpg"%(model_name, str("mean"))), dpi = 600)
    plt.close()

# %%
# log_dict = model_dict["softmax"]

# for i_graph in graph_name_list:
#     log_list_temp = np.array(log_dict[i_graph])
#     num_langs_traj = log_list_temp[:,:,[0,-1]]
#     non_nan_idx = ~np.isnan(num_langs_traj[0,:,-1])
#     num_langs_traj = num_langs_traj[:,non_nan_idx,:]
    
#     num_langs_traj = np.mean(num_langs_traj, axis=0)
#     print(i_graph, num_langs_traj[-1,-1])



