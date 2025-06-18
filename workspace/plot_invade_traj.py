# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path
import os
import pickle

BASE_PATH = Path("/home/zihangw/EvoComm")
plt.rcParams.update({'font.size': 20})

# %%
param_file = BASE_PATH / "param_space" / "invade_param_demes_multi_logger.txt"
out_path_base = BASE_PATH / "results_invade_logger"
# combined_name = BASE_PATH / "results_invade_combined" / "invade_param_demes_multi.csv"
# graph_info_path = BASE_PATH / "results_invade_combined" / "invade_graph_info.csv"

with open(param_file, "r") as f:
    param_sim = f.readlines()
param_sim = [x.strip() for x in param_sim]
param_sim = [x.split(" ") for x in param_sim]

# %%
# df_all = pd.DataFrame()
# param = param_sim[-1]
for param in param_sim:
    graph_path = param[2]

    graph_base = os.path.dirname(graph_path)
    graph_name = os.path.basename(graph_path).split(".")[0]
    out_path = out_path_base / graph_base / graph_name
    
    num_demes = int(graph_name.split("_")[1])
    num_edge_added = int(graph_name.split("_")[2])
    beta = int(graph_name.split("_")[3].split("m")[-1])

    if (num_demes != 50) or (num_edge_added not in [5,10]):
        continue

    if not out_path.exists():
        print(f"Path {out_path} does not exist, skipping.")
        continue

    files = os.listdir(out_path)
    for file in files:
        file_name = file.split(".")[0]
        file_ext = file.split(".")[-1]
        if file_ext != "pkl":
            continue
        
        with open(out_path / file, "rb") as f:
            logger = pickle.load(f)

        df = pd.DataFrame(logger)

        fig, ax = plt.subplots(figsize=(8, 6))
        plt.title(f"number of demes: {num_demes}, number of edges added: {num_edge_added}")
        plt.plot(df["iteration"], df["num_group_0"], label="wild type")
        plt.plot(df["iteration"], df["num_group_1"], label="invador")
        plt.xlabel("Iteration")
        plt.ylabel("Number of Individuals")
        plt.legend()
        fig.savefig(out_path / f"{file_name}.jpg", dpi = 300, bbox_inches='tight')
        plt.close(fig)
# df_list = [pd.read_csv(out_path / file, sep="\t") for file in files]

# %%
