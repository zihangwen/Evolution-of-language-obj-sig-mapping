# %%
import numpy as np
import pandas as pd
from dataclasses import dataclass
import copy
import time
import os
import sys
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

BASE_PATH = Path("/home/zihangw/EvoComm")

# %%
# param_file = BASE_PATH / "param_space" / "invade_real_clean.txt"
# out_path_base = BASE_PATH / "results_invade"
combined_name = BASE_PATH / "results_invade_combined" / "invade_real_clean.csv"
graph_info_path = BASE_PATH / "results_invade_combined" / "invade_real_clean_info.csv"

df_result = pd.read_csv(combined_name, sep="\t")
df_info = pd.read_csv(graph_info_path, sep="\t")
df_all = pd.merge(df_result, df_info, on="graph_name")

df_all["fixation_count_modified"] = df_all["fixation_count"] + df_all["co_existence_count"]
df_all["pfix_modified"] = df_all["fixation_count_modified"] / df_all["num_trials"]

# %%
time_mapping = {
    "hg": 1,
    "village_all": 2,
    "padgett_florence_families": 3,
    "email_eu": 4,
    "facebook": 5,
    "retweet_copen": 6
}

def map_time(name):
    for key, value in time_mapping.items():
        if key in name:  # substring match
            return value
    return None  # default if no match

df_all["time_category"] = df_all["graph_name"].apply(map_time)
df_all = df_all[~df_all["graph_name"].str.contains("padgett_florence_families|email_eu")]
df_all["pfix * num_nodes"] = df_all["pfix_modified"] * df_all["num_nodes"]


# %%
sns.lineplot(data=df_all, x="time_category", y="num_nodes", marker="o")
plt.title("Average Number of Nodes over Time")
plt.xticks(ticks=list(time_mapping.values()), labels=list(time_mapping.keys()), rotation=45)
plt.show()

# %%
sns.lineplot(data=df_all, x="time_category", y="num_edges", marker="o", color="orange")
plt.title("Average Number of Edges over Time")
plt.xticks(ticks=list(time_mapping.values()), labels=list(time_mapping.keys()), rotation=45)
plt.show()

# %%
sns.lineplot(data=df_all, x="time_category", y="graph_mean_degree", marker="o", color="green")
plt.title("Average Mean Degree over Time")
plt.xticks(ticks=list(time_mapping.values()), labels=list(time_mapping.keys()), rotation=45)
plt.show()

# %%
sns.lineplot(data=df_all, x="time_category", y="average_clustering", marker="o", color="red")
plt.title("Average Clustering Coefficient over Time")
plt.xticks(ticks=list(time_mapping.values()), labels=list(time_mapping.keys()), rotation=45)
plt.show()

# %%
sns.lineplot(data=df_all, x="time_category", y="assortativity", marker="o", color="purple")
plt.title("Average Assortativity over Time")
plt.xticks(ticks=list(time_mapping.values()), labels=list(time_mapping.keys()), rotation=45)
plt.show()

# %%
sns.lineplot(data=df_all, x="time_category", y="diameter", marker="o", color="brown")
plt.title("Average Diameter over Time")
plt.xticks(ticks=list(time_mapping.values()), labels=list(time_mapping.keys()), rotation=45)
plt.show()

# %%
sns.lineplot(data=df_all, x="time_category", y="modularity", marker="o", color="cyan")
plt.title("Average Modularity over Time")
plt.xticks(ticks=list(time_mapping.values()), labels=list(time_mapping.keys()), rotation=45)
plt.show()

# %%
sns.lineplot(data=df_all, x="time_category", y="transitivity", marker="o", color="magenta")
plt.title("Average Transitivity over Time")
plt.xticks(ticks=list(time_mapping.values()), labels=list(time_mapping.keys()), rotation=45)
plt.show()

# %%
sns.lineplot(data=df_all, x="time_category", y="algebraic_connectivity", marker="o", color="lime")
plt.title("Average Algebraic Connectivity over Time")
plt.xticks(ticks=list(time_mapping.values()), labels=list(time_mapping.keys()), rotation=45)
plt.show()

# %%
sns.lineplot(data=df_all, x="time_category", y="pfix_modified", marker="o", color="darkorange")
plt.title("Average Modified Fixation Probability over Time")
plt.xticks(ticks=list(time_mapping.values()), labels=list(time_mapping.keys()), rotation=45)
plt.show()

# %%
sns.lineplot(data=df_all, x="time_category", y="fixation_time", marker="o", color="grey")
plt.title("Average Fixation Time over Time")
plt.xticks(ticks=list(time_mapping.values()), labels=list(time_mapping.keys()), rotation=45)
plt.show()

# %%
sns.lineplot(data=df_all, x="time_category", y="pfix * num_nodes", marker="o", color="olive")
plt.title("Average Normalized Fixation Probability over Time")
plt.xticks(ticks=list(time_mapping.values()), labels=list(time_mapping.keys()), rotation=45)
plt.show()

# %%
sns.lineplot(data=df_all, x="algebraic_connectivity", y="pfix * num_nodes", marker="o", color="teal")
plt.title("Normalized Fixation Probability vs. Algebraic Connectivity")
plt.show()

# %%
sns.lineplot(data=df_all, x="transitivity", y="pfix * num_nodes", marker="o", color="navy")
plt.title("Normalized Fixation Probability vs. Transitivity")
plt.show()
