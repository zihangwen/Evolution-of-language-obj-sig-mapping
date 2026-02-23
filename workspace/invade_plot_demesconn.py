# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path

BASE_PATH = Path("/home/zihangw/EvoComm")
# plt.rcParams.update({'font.size': 20})

# %%
combined_name = BASE_PATH / "results_cpp_combined" / "param_demesconn_edges_invade.csv"
graph_info_path = BASE_PATH / "results_cpp_combined" / "param_demesconn_edges_info.csv"

df = pd.read_csv(combined_name, sep="\t")
df_info = pd.read_csv(graph_info_path, sep="\t")
df_all = pd.merge(df, df_info, on="graph_name")

df_all["fixation_count_modified"] = df_all["fixation_count"] + df_all["co_existence_count"]
df_all["pfix_modified"] = df_all["fixation_count_modified"] / df_all["num_trials"]

# %% supp
combined_name = BASE_PATH / "results_cpp_combined" / "param_demesconn_edges_invade_supp.csv"
graph_info_path = BASE_PATH / "results_cpp_combined" / "param_demesconn_edges_info_supp.csv"

df = pd.read_csv(combined_name, sep="\t")
df_info = pd.read_csv(graph_info_path, sep="\t")
df_supp = pd.merge(df, df_info, on="graph_name")

df_supp["fixation_count_modified"] = df_supp["fixation_count"] + df_supp["co_existence_count"]
df_supp["pfix_modified"] = df_supp["fixation_count_modified"] / df_supp["num_trials"]

df_all = pd.concat([df_all, df_supp], ignore_index=True)

df_all.sort_values(by=["num_demes", "num_edge_added", "demeconn_number"], inplace=True, ignore_index=True)

# %%
df_select = df_all[(df_all["num_demes"] == 20) & (df_all["num_edge_added"] == 1)]
df_select_star = df_select[df_select["demeconn_number"] == "star"]
df_select = df_select[df_select["demeconn_number"] != "star"]
sns.lineplot(data=df_select, x="demeconn_number", y="pfix", marker="o")
plt.axhline(y=df_select_star["pfix"].values[0], color="purple", linestyle="--", label="star topology")
plt.title("Probability of Fixation vs. High level Deme Connections")
plt.xlabel(r"$d_{deme}$")
plt.legend()

# %%
