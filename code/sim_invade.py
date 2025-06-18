import numpy as np
from dataclasses import dataclass
import copy
import time
import os
import sys
from pathlib import Path

from utilities import *
from model import Config, LanguageModelStabilized
from simulations import SimulationGraphInvade


if __name__ == "__main__":
    num_objects = int(sys.argv[1])
    num_sounds = int(sys.argv[2])
    graph_path = sys.argv[3]
    num_trials = int(sys.argv[4])
    out_path_base = sys.argv[5]
    n_trial = int(sys.argv[6])

    # num_objects = 5
    # num_sounds = 5
    # graph_path = "networks/toy/star_10.txt"
    # num_trials = 10
    # out_path_base = "results_test"
    # n_trial = 0

    graph_base = os.path.dirname(graph_path)
    graph_name = os.path.basename(graph_path).split(".")[0]

    out_path = os.path.join(out_path_base, graph_base, graph_name)
    os.makedirs(out_path, exist_ok=True)

    config = Config(num_objects, num_sounds)

    lang_init = LanguageModelStabilized(0, num_objects, num_sounds)
    # lang_init.initialize_language()
    lang_init.P = np.array([[1., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0.],
                            [1., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0.],
                            [0., 0., 0., 1., 0.]])
    lang_init.Q = NormalizeEPS(lang_init.P.T)
    # lang_init.Q = np.array([[0.5, 0., 0.5, 0., 0.],
    #                         [0., 0.5, 0., 0.5, 0.],
    #                         [0.2, 0.2, 0.2, 0.2, 0.2],
    #                         [0., 0., 0., 0., 1.],
    #                         [0.2, 0.2, 0.2, 0.2, 0.2]])

    lang_invade = LanguageModelStabilized(1, num_objects, num_sounds)
    # lang_invade.initialize_language()
    lang_invade.P = np.array([[0., 0., 1., 0., 0.],
                              [0., 0., 0., 1., 0.],
                              [1., 0., 0., 0., 0.],
                              [0., 1., 0., 0., 0.],
                              [1., 0., 0., 0., 0.]])
    lang_invade.Q = NormalizeEPS(lang_invade.P.T)
    # lang_invade.Q = np.array([[0. , 0. , 0.5, 0. , 0.5],
    #                           [0., 0., 0., 1., 0.],
    #                           [1., 0., 0., 0., 0.],
    #                           [0., 1., 0., 0., 0.],
    #                           [0.2, 0.2, 0.2, 0.2]])
    # array([0., 0., 1., 0., 0.]), array([0., 0., 0., 1., 0.]), array([1., 0., 0., 0., 0.]), array([0., 1., 0., 0., 0.]), array([1., 0., 0., 0., 0.])
    # [array([0. , 0. , 0.5, 0. , 0.5]), array([0., 0., 0., 1., 0.]), array([1., 0., 0., 0., 0.]), array([0., 1., 0., 0., 0.]), array([0.2, 0.2, 0.2, 0.2, 0.2])]

    fixation_time = 0
    co_existence_count = 0
    fixation_count = 0
    for i_trial in range(num_trials):
        time_start = time.time()
        sim = SimulationGraphInvade(config, graph_path)
        sim.initialize(LanguageModelStabilized, lang_init, lang_invade)
        result, i_t = sim.run(int(1e7))
        time_end = time.time()
        print("graph: %s, trial: %d, time cost: %.2f" % (graph_name, i_trial, time_end - time_start))

        if result == "fix":
            fixation_time = fixation_time * (fixation_count / (fixation_count + 1)) + i_t / (fixation_count + 1)
            fixation_count += 1
        elif result == "coexist":
            co_existence_count += 1
            sim.logger.save_logs(Path(out_path), f"{n_trial}_{i_trial}_logger.pkl")

    with open(os.path.join(out_path, f"{n_trial}.txt"), "w") as f:
        f.write("# graph_name\t")
        f.write("num_trials\t")
        f.write("fixation_count\t")
        f.write("fixation_time\t")
        f.write("co_existence_count\n")

        f.write(f"{graph_name}\t")
        f.write(f"{num_trials}\t")
        f.write(f"{fixation_count}\t")
        f.write(f"{fixation_time}\t")
        f.write(f"{co_existence_count}\n")
        # np.savetxt(os.path.join(out_path, f"{n_trial*num_trials+i_trial}.txt"), logger, fmt='%g')

    print("hello world!")