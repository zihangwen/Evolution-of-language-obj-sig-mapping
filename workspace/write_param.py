# %%
import numpy as np
import os
import re
from pathlib import Path

BASE_PATH = Path("/home/zihangw/EvoComm/")
# NET_PATH = BASE_PATH / "networks"

# %%
# if __name__ == "__main__":
# num_objects = 5
# num_sounds = 5
# num_languages = 100
# model_name_list = ["norm", "softmax"]
# graph_path_list = ["graphs/3-r.txt", "graphs/5-r.txt", "graphs/detour.txt", "graphs/star.txt", "graphs/wheel.txt", "graphs/wm_100.txt"]
# # graph_path_list = ["graphs/detour.txt", "graphs/wm_100.txt"]
# num_runs = int(1e5)
# out_path_base = "results"
# # n_trial = range(10)

# with open("param_agent.txt", "w") as f:
#     for model_name in model_name_list:
#         for graph_path in graph_path_list:
#             f.write("%d %d %d %s %s %d %s\n"%(num_objects, num_sounds, num_languages, model_name, graph_path, num_runs, out_path_base))

# num_objects = 5
# num_sounds = 5
# num_languages = 10
# model_name_list = ["norm", "softmax"]
# num_runs = int(25_000)

# graph_path_list = ["networks/toy_star/" + a for a in os.listdir("/home/zihangw/EvoComm/networks/toy_star")]
# out_path_base = "results/toy_star"
# # n_trial = range(10)

# with open("param_toy_regular.txt", "a") as f:
#     for model_name in model_name_list:
#         for graph_path in graph_path_list:
#             f.write("%d %d %d %s %s %d %s\n"%(num_objects, num_sounds, num_languages, model_name, graph_path, num_runs, out_path_base))

# num_languages = 100
# num_runs = int(50_000)
# graph_path_list = ["networks/regular_100_4_triangle/reg_tri_100_4_%d.txt" %i for i in range(0,100,5)]
# out_path_base = "results/regular_100_4_triangle"
# # n_trial = range(10)

# with open("param_toy_regular.txt", "a") as f:
#     for model_name in model_name_list:
#         for graph_path in graph_path_list:
#             f.write("%d %d %d %s %s %d %s\n"%(num_objects, num_sounds, num_languages, model_name, graph_path, num_runs, out_path_base))

# num_objects = 5
# num_sounds = 5
# model_name_list = ["norm", "softmax"]
# graph_folder_name = "bottleneck_demes"
# graph_path_list = [f"networks/{graph_folder_name}/" + a for a in os.listdir(f"/home/zihangw/EvoComm/networks/{graph_folder_name}")]
# num_runs = int(100_000)
# out_path_base = f"results/{graph_folder_name}"
# # n_trial = range(10)
# sample_times = 100
# temperature = 1.0
# with open("../param_space/param_bottleneck_demes.txt", "w") as f:
#     for model_name in model_name_list:
#         for graph_path in graph_path_list:
#             f.write(f"{num_objects} {num_sounds} {model_name} {graph_path} {num_runs} {out_path_base} {sample_times} {temperature:.1f}\n")

# num_objects = 5
# num_sounds = 5
# model_name_list = ["norm", "softmax"]
# sample_times_list = [1, 5, 10, 20, 40, 100]
# temperature_list = [1.0, 2.0, 5.0, 10.0]

# graph_folder_name = "temp_sample_size"
# graph_path_list = [f"networks/{graph_folder_name}/" + a for a in os.listdir(f"/home/zihangw/EvoComm/networks/{graph_folder_name}")]
# num_runs = int(50_000)
# out_path_base = f"results/{graph_folder_name}"
# # n_trial = range(10)

# with open(f"../param_space/param_{graph_folder_name}.txt", "w") as f:
#     for model_name in model_name_list:
#         for graph_path in graph_path_list:
#             for sample_times in sample_times_list:
#                 for temperature in temperature_list:
#                     f.write(f"{num_objects} {num_sounds} {model_name} {graph_path} {num_runs} {out_path_base} {sample_times} {temperature}\n")
#                     # f.write("%d %d %s %s %d %s\n"%(num_objects, num_sounds, model_name, graph_path, num_runs, out_path_base))

# ----- param_bottleneck_demes_invade ----- #
# num_objects = 5
# num_sounds = 5
# graph_folder_name = "bottleneck_demes"
# graph_path_list = [f"networks/{graph_folder_name}/" + a for a in os.listdir(f"/home/zihangw/EvoComm/networks/{graph_folder_name}")]

# graph_folder_name = "graphs"
# graph_path_list += [f"networks/{graph_folder_name}/" + a for a in os.listdir(f"/home/zihangw/EvoComm/networks/{graph_folder_name}")]

# num_trials = int(10_000)
# with open("../param_space/param_bottleneck_demes_invade.txt", "w") as f:
#         for graph_path in graph_path_list:
#             f.write(f"{num_objects} {num_sounds} {graph_path} {num_trials}\n")

# ----- param_PA_invade ----- #
# num_objects = 5
# num_sounds = 5
# graph_folder_name = "PA_100"
# graph_path_list = [f"networks/{graph_folder_name}/" + a for a in os.listdir(f"/home/zihangw/EvoComm/networks/{graph_folder_name}")]

# graph_folder_name = "social_network"
# graph_path_list += [f"networks/{graph_folder_name}/" + a for a in os.listdir(f"/home/zihangw/EvoComm/networks/{graph_folder_name}")]

# num_trials = int(10_000)
# with open("../param_space/param_PA_invade.txt", "w") as f:
#         for graph_path in graph_path_list:
#             f.write(f"{num_objects} {num_sounds} {graph_path} {num_trials}\n")

# ----- param_regular_invade ----- #
# num_objects = 5
# num_sounds = 5
# graph_folder_name = "paper_regular_100"
# graph_path_list = [f"networks/{graph_folder_name}/" + a for a in os.listdir(f"/home/zihangw/EvoComm/networks/{graph_folder_name}")]
# graph_path_list = sorted(graph_path_list, key=lambda s: tuple(map(int, re.findall(r"\d+", s))))

# # graph_path_list = graph_path_list[::5]

# num_trials = int(10_000)
# with open("../param_space/param_regular_invade.txt", "w") as f:
#     for graph_path in graph_path_list:
#         f.write(f"{num_objects} {num_sounds} {graph_path} {num_trials}\n")

# %%
# ----- param_regular_4_invade ----- #
# num_objects = 5
# num_sounds = 5
# graph_folder_name = "regular_100_4_triangle"
# graph_path_list = [f"networks/{graph_folder_name}/" + a for a in os.listdir(f"/home/zihangw/EvoComm/networks/{graph_folder_name}")]
# graph_path_list = sorted(graph_path_list, key=lambda s: tuple(map(int, re.findall(r"\d+", s))))

# graph_path_list = graph_path_list[::5]

# num_trials = int(10_000)
# with open("../param_space/param_regular_4_invade.txt", "w") as f:
#     for graph_path in graph_path_list:
#         f.write(f"{num_objects} {num_sounds} {graph_path} {num_trials}\n")

# %%
# ----- deme model invasion ----- #
# num_objects = 5
# num_sounds = 5
# # graph_folder_list = [f"bottleneck_demes_{num_demes}" for num_demes in [5, 10, 20, 50, 100]]
# graph_folder_list = [f"bottleneck_demes_{num_demes}" for num_demes in [20, 50]]

# num_trials = int(10_000)
# with open(BASE_PATH / "param_space" / "invade_param_demes_multi_logger.txt", "w") as f:
#     for graph_folder_name in graph_folder_list:
#         graph_path_list = [
#             f"networks/{graph_folder_name}/" + a for a in os.listdir(BASE_PATH / "networks" / graph_folder_name)
#         ]
#         graph_path_list = sorted(graph_path_list, key=lambda s: tuple(map(int, re.findall(r"\d+", s))))
                
#         for graph_path in graph_path_list:
#             f.write(f"{num_objects} {num_sounds} {graph_path} {num_trials}\n")

# %%
# ----- deme model invasion (deme size and deme number) ----- #
# deme_size_list = [5, 10, 20]
# # num_demes_list = [5, 10, 20, 50]
# num_demes_list = [1]

# num_objects = 5
# num_sounds = 5
# graph_folder_list = [
#     f"bottleneck_demes{num_demes}_size{deme_size}" for num_demes in num_demes_list for deme_size in deme_size_list
# ]

# num_trials = int(10_000)
# with open(BASE_PATH / "param_space" / "invade_param_demes_multi_ns_1.txt", "w") as f:
#     for graph_folder_name in graph_folder_list:
#         graph_path_list = [
#             f"networks/{graph_folder_name}/" + a for a in os.listdir(BASE_PATH / "networks" / graph_folder_name)
#         ]
#         graph_path_list = sorted(graph_path_list, key=lambda s: tuple(map(int, re.findall(r"\d+", s))))
                
#         for graph_path in graph_path_list:
#             f.write(f"{num_objects} {num_sounds} {graph_path} {num_trials}\n")

# %%
# ----- deme model evolve (deme size and deme number) ----- #
# deme_size_list = [5, 10, 20]
# num_demes_list = [1, 5, 10, 20, 50]

# num_objects = 5
# num_sounds = 5

# model_name_list = ["norm"]
# sample_times_list = [10]
# temperature_list = [1.0]

# num_runs = int(50_000)

# graph_folder_list = [
#     f"bottleneck_demes{num_demes}_size{deme_size}" for num_demes in num_demes_list for deme_size in deme_size_list
# ]

# num_trials = int(10_000)
# with open(BASE_PATH / "param_space" / "evolve_param_demes_multi_ns.txt", "w") as f:
#     for graph_folder_name in graph_folder_list:
#         out_path_base = f"results/{graph_folder_name}"
#         graph_path_list = [
#             f"networks/{graph_folder_name}/" + a for a in os.listdir(BASE_PATH / "networks" / graph_folder_name)
#         ]
#         graph_path_list = sorted(graph_path_list, key=lambda s: tuple(map(int, re.findall(r"\d+", s))))
#         for graph_path in graph_path_list:
#             for model_name in model_name_list:
#                 for sample_times in sample_times_list:
#                     for temperature in temperature_list:
#                         f.write(f"{num_objects} {num_sounds} {model_name} {graph_path} {num_runs} {sample_times} {temperature}\n")

# %%
# ----- deme model invasion (fix pop size) ----- #
# num_objects = 5
# num_sounds = 5

# num_trials = int(10_000)

# # graph_folder_list = [
# #     "wm_100",
# #     "wm_1000",
# #     "bottleneck_pop100",
# #     "bottleneck_pop1000",
# # ]

# graph_folder_list = [
#     "wm_500",
#     "bottleneck_pop500",
# ]

# with open(BASE_PATH / "param_space" / "invade_param_demes_500_popsize.txt", "w") as f:
#     for graph_folder_name in graph_folder_list:
#         graph_path_list = [
#             f"networks/{graph_folder_name}/" + a for a in os.listdir(BASE_PATH / "networks" / graph_folder_name)
#         ]
#         graph_path_list = sorted(graph_path_list, key=lambda s: tuple(map(int, re.findall(r"\d+", s))))
                
#         for graph_path in graph_path_list:
#             f.write(f"{num_objects} {num_sounds} {graph_path} {num_trials}\n")

# %%
# ----- deme model evolve (fix pop size) ----- #
# num_objects = 5
# num_sounds = 5

# model_name_list = ["norm"]
# sample_times_list = [10]
# temperature_list = [1.0]

# iteration = int(50_000)

# graph_folder_list = [
#     "wm_500",
#     "bottleneck_pop500",
# ]

# with open(BASE_PATH / "param_space" / "evolve_param_demes_500_popsize.txt", "w") as f:
#     for graph_folder_name in graph_folder_list:
#         out_path_base = f"results/{graph_folder_name}"
#         graph_path_list = [
#             f"networks/{graph_folder_name}/" + a for a in os.listdir(BASE_PATH / "networks" / graph_folder_name)
#         ]
#         graph_path_list = sorted(graph_path_list, key=lambda s: tuple(map(int, re.findall(r"\d+", s))))
#         for graph_path in graph_path_list:
#             for model_name in model_name_list:
#                 for sample_times in sample_times_list:
#                     for temperature in temperature_list:
#                         f.write(f"{num_objects} {num_sounds} {model_name} {graph_path} {iteration} {sample_times} {temperature}\n")

# %%
# ----- deme model invasion (fix pop size) ----- #
num_objects = 5
num_sounds = 5

num_trials = int(10_000)

# graph_folder_list = [
#     "wm_100",
#     "wm_1000",
#     "bottleneck_pop100",
#     "bottleneck_pop1000",
# ]

graph_folder_list = [
    "cleaned",
]

with open(BASE_PATH / "param_space" / "invade_real_clean.txt", "w") as f:
    for graph_folder_name in graph_folder_list:
        graph_path_list = [
            f"real_data/{graph_folder_name}/" + a for a in os.listdir(BASE_PATH / "real_data" / graph_folder_name)
        ]
        graph_path_list = sorted(graph_path_list, key=lambda s: tuple(map(int, re.findall(r"\d+", s))))
                
        for graph_path in graph_path_list:
            f.write(f"{num_objects} {num_sounds} {graph_path} {num_trials}\n")

# %%
# ----- deme model evolve (fix pop size) ----- #
num_objects = 5
num_sounds = 5

model_name_list = ["norm"]
sample_times_list = [10]
temperature_list = [1.0]

iteration = int(50_000)

graph_folder_list = [
    "cleaned",
]

with open(BASE_PATH / "param_space" / "evolve_real_clean.txt", "w") as f:
    for graph_folder_name in graph_folder_list:
        out_path_base = f"results/{graph_folder_name}"
        graph_path_list = [
            f"real_data/{graph_folder_name}/" + a for a in os.listdir(BASE_PATH / "real_data" / graph_folder_name)
        ]
        graph_path_list = sorted(graph_path_list, key=lambda s: tuple(map(int, re.findall(r"\d+", s))))
        for graph_path in graph_path_list:
            for model_name in model_name_list:
                for sample_times in sample_times_list:
                    for temperature in temperature_list:
                        f.write(f"{num_objects} {num_sounds} {model_name} {graph_path} {iteration} {sample_times} {temperature}\n")


# %%
# num_objects=$(sed -n "${line}p" ${param_file} | awk '{print $1}')
# num_sounds=$(sed -n "${line}p" ${param_file} | awk '{print $2}')
# graph_path=$(sed -n "${line}p" ${param_file} | awk '{print $4}')
# num_trials=$(sed -n "${line}p" ${param_file} | awk '{print $5}')

# %%
