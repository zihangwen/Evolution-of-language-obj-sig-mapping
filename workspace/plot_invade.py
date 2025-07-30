# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path

BASE_PATH = Path("/home/zihangw/EvoComm")
plt.rcParams.update({'font.size': 20})

# sns.color_palette("tab10")

# %%
combined_name = BASE_PATH / "results_invade_combined" / "invade_param_demes_multi_ns.csv"
df = pd.read_csv(combined_name, sep="\t")

# %% df select
df = df[df["num_edge_added"].isin([10])]
# df = df[df["num_edge_added"] != 1]

# %%
# df = df[df["num_edge_added"] >= 10]
fig, axe = plt.subplots(figsize=(8, 6))
sns.lineplot(data=df, x="num_demes", y="pfix", hue="deme_size", marker='o', errorbar=("sd"), palette="tab10")
# plt.title(r"Probability of Fixation vs Number of Edges Added ($N_T = 1000$)")
plt.xlabel("Number of demes")
plt.ylabel("Probability of Fixation")
plt.tight_layout()
# axe.get_legend().remove()
# plt.show()
fig.savefig(
    BASE_PATH / "figures" / "invade_pfix_vs_n_demes.jpg",
    dpi=600
)

# %%
fig, axe = plt.subplots(figsize=(8, 6))
sns.lineplot(data=df, x="deme_size", y="pfix", hue="num_demes", marker='o', errorbar=("sd"), palette="tab10")
# plt.title(r"Probability of Fixation vs Number of Edges Added ($N_T = 1000$)")
plt.xlabel("deme size")
plt.ylabel("Probability of Fixation")
plt.tight_layout()
# axe.get_legend().remove()
# plt.show()
fig.savefig(
    BASE_PATH / "figures" / "invade_pfix_vs_deme_size.jpg",
    dpi=600
)

# %%
fig, axe = plt.subplots(figsize=(8, 6))
sns.lineplot(data=df, x="num_demes", y="fixation_time", hue="deme_size", marker='o', errorbar=("sd"), palette="tab10")
# plt.title(r"Fixation Time vs Number of Edges Added ($N_T = 1000$)")
plt.xlabel("Number of demes")
plt.ylabel("Fixation Time | Fixation")
plt.tight_layout()
# axe.get_legend().remove()
plt.yscale('log')
# plt.show()
fig.savefig(
    BASE_PATH / "figures" / "invade_fixation_time_vs_n_demes.jpg",
    dpi=600
)

# %%
fig, axe = plt.subplots(figsize=(8, 6))
sns.lineplot(data=df, x="deme_size", y="fixation_time", hue="num_demes", marker='o', errorbar=("sd"), palette="tab10")
# plt.title(r"Fixation Time vs Number of Edges Added ($N_T = 1000$)")
plt.xlabel("deme size")
plt.ylabel("Fixation Time | Fixation")
plt.tight_layout()
# axe.get_legend().remove()
plt.yscale('log')
# plt.show()
fig.savefig(
    BASE_PATH / "figures" / "invade_fixation_time_vs_demesize.jpg",
    dpi=600
)

# %%
# fig, axe = plt.subplots(figsize=(8, 6))
# sns.lineplot(data=df, x="num_edge_added", y="pco_exist", hue="num_demes", marker='o', errorbar=("sd"), palette="tab10")
# # plt.title(r"Probability of Co-existence vs Number of Edges Added ($N_T = 1000$)")
# plt.xlabel("Number of Edges Added")
# plt.ylabel("Probability of Co-existence")
# axe.get_legend().remove()
# plt.tight_layout()
# # plt.show()
# fig.savefig(
#     BASE_PATH / "figures" / "invade_pco_exist_vs_num_edge_added_fast.jpg",
#     dpi=600
# )

# %%
