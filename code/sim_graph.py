import numpy as np
from dataclasses import dataclass
import copy
import time
import os
import sys
from utilities import *
from model import Config, LanguageModelNorm, LanguageModelSoftmax
from simulations import SimulationGraph


if __name__ == "__main__":
    num_objects = int(sys.argv[1])
    num_sounds = int(sys.argv[2])
    model_name = sys.argv[3]
    graph_path = sys.argv[4]
    num_runs = int(sys.argv[5])
    out_path_base = sys.argv[6]
    sample_times = int(sys.argv[7])
    temperature = float(sys.argv[8])
    n_trial = int(sys.argv[9])
    # num_objects = 5
    # num_sounds = 5
    # num_languages = 10
    # model_name = "softmax"
    # graph_path = "networks/toy/star_10.txt"
    # num_runs = 10000
    # out_path_base = "results_test"
    # n_trial = 0

    num_trials = 10
    config = Config(num_objects, num_sounds,
                    sample_times = sample_times, temperature = temperature)
    if model_name == "norm":
        LangModel = LanguageModelNorm
    elif model_name == "softmax":
        LangModel = LanguageModelSoftmax

    for i_trial in range(num_trials):
        time_start = time.time()
        
        graph_name = os.path.basename(graph_path).split(".")[0]

        sim = SimulationGraph(config, graph_path, LangModel)
        sim.run(num_runs)
        time_end = time.time()
        print("graph: %s, trial: %d, time cost: %.2f" % (graph_name, i_trial, time_end - time_start))

        logger = np.array(sim.get_logger())
        out_path = os.path.join(out_path_base, model_name, graph_name)
        os.makedirs(out_path, exist_ok=True)
        np.savetxt(os.path.join(out_path, f"st_{sample_times}_temp_{temperature:.1f}_{n_trial*num_trials+i_trial}.txt"), logger, fmt='%g')

    print("hello world!")