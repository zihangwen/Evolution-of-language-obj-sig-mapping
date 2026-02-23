# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path

BASE_PATH = Path("/home/zihangw/EvoComm")
# plt.rcParams.update({'font.size': 20})

# %%
combined_name = BASE_PATH / "results_cpp_combined" / "param_demesconn_edges_evolve.csv"
graph_info_path = BASE_PATH / "results_cpp_combined" / "param_demesconn_edges_info.csv"

df = pd.read_csv(combined_name, sep="\t")
df_info = pd.read_csv(graph_info_path, sep="\t")
df_all = pd.merge(df, df_info, on="graph_name")

# %% supp
combined_name = BASE_PATH / "results_cpp_combined" / "param_demesconn_edges_evolve_supp.csv"
graph_info_path = BASE_PATH / "results_cpp_combined" / "param_demesconn_edges_info_supp.csv"

df = pd.read_csv(combined_name, sep="\t")
df_info = pd.read_csv(graph_info_path, sep="\t")
df_supp = pd.merge(df, df_info, on="graph_name")

df_all = pd.concat([df_all, df_supp], ignore_index=True)
df_all.sort_values(by=["num_demes", "num_edge_added", "demeconn_number"], inplace=True, ignore_index=True)

# %%
df_select = df_all[(df_all["num_demes"] == 10) & (df_all["num_edge_added"] == 1)]
df_select_star = df_select[df_select["demeconn_number"] == "star"]
df_select = df_select[df_select["demeconn_number"] != "star"]
sns.lineplot(data=df_select, x="demeconn_number", y="final_num_lang",marker="o")
plt.axhline(y=df_select_star["final_num_lang"].values[0], color="purple", linestyle="--", label="star topology")
plt.title("Average Final Mutant Frequency vs. High level Deme Connections")
plt.xlabel(r"$d_{deme}$")
plt.ylabel("Average Final Number of Languages")
plt.legend()

# %% num_edge_added
# df_select = df_all[(df_all["num_demes"] == 10)]
# df_select_star = df_select[df_select["demeconn_number"] == "star"]
# df_select = df_select[df_select["demeconn_number"] != "star"]
# sns.lineplot(data=df_select, x="demeconn_number", y="final_num_lang", hue="num_edge_added", marker="o")
# plt.axhline(y=df_select_star["final_num_lang"].values[0], color="purple", linestyle="--", label="star topology")
# plt.title("Average Final Mutant Frequency vs. High level Deme Connections")
# plt.xlabel(r"$d_{deme}$")
# plt.ylabel("Average Final Number of Languages")
# plt.legend()

# %% num_demes
# df_select = df_all[(df_all["num_edge_added"] == 1)]
# df_select_star = df_select[df_select["demeconn_number"] == "star"]
# df_select = df_select[df_select["demeconn_number"] != "star"]
# sns.lineplot(data=df_select, x="demeconn_number", y="final_num_lang", hue="num_demes", marker="o")
# plt.axhline(y=df_select_star["final_num_lang"].values[0], color="purple", linestyle="--", label="star topology")
# plt.title("Average Final Mutant Frequency vs. High level Deme Connections")
# plt.xlabel(r"$d_{deme}$")
# plt.ylabel("Average Final Number of Languages")
# plt.legend()

# %%