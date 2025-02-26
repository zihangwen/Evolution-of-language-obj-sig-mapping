import numpy as np
import os


if __name__ == "__main__":
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
    # num_languages = 100
    # model_name_list = ["norm", "softmax"]
    # graph_folder_name = "bottleneck_demes"
    # graph_path_list = [f"networks/{graph_folder_name}/" + a for a in os.listdir(f"/home/zihangw/EvoComm/networks/{graph_folder_name}")]
    # num_runs = int(50_000)
    # out_path_base = f"results/{graph_folder_name}"
    # # n_trial = range(10)
    
    # with open("../param_space/param_bottleneck_demes.txt", "w") as f:
    #     for model_name in model_name_list:
    #         for graph_path in graph_path_list:
    #             f.write("%d %d %d %s %s %d %s\n"%(num_objects, num_sounds, num_languages, model_name, graph_path, num_runs, out_path_base))

    num_objects = 5
    num_sounds = 5
    model_name_list = ["norm", "softmax"]
    sample_times_list = [1, 5, 10, 20, 40, 100]
    temperature_list = [0.1, 0.5, 1, 2, 5, 10]

    graph_folder_name = "temp_sample_size"
    graph_path_list = [f"networks/{graph_folder_name}/" + a for a in os.listdir(f"/home/zihangw/EvoComm/networks/{graph_folder_name}")]
    num_runs = int(50_000)
    out_path_base = f"results/{graph_folder_name}"
    # n_trial = range(10)
    
    with open(f"../param_space/param_{graph_folder_name}.txt", "w") as f:
        for model_name in model_name_list:
            for graph_path in graph_path_list:
                for sample_times in sample_times_list:
                    for temperature in temperature_list:
                        f.write(f"{num_objects} {num_sounds} {model_name} {graph_path} {num_runs} {out_path_base} {sample_times} {temperature}\n")
                        # f.write("%d %d %s %s %d %s\n"%(num_objects, num_sounds, model_name, graph_path, num_runs, out_path_base))

