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
combined_name = BASE_PATH / "results_invade_combined" / "invade_param_demes_fix_popsize.csv"
df = pd.read_csv(combined_name, sep="\t")

# %% df select
df_wm = df[df["graph_base"] == "wm"]
df_select = df[df["graph_base"] != "wm"]
# df = df[df["num_edge_added"].isin([10])]
# df = df[df["num_edge_added"] != 1]

# %%
name_mapping_dict = {
    "num_demes": "Number of Demes",
    "deme_size": "Deme Size",
    "pop_size": "Total Population Size",
    "num_edge_added": "Number of Edges Added Between Each Two Demes",
    "beta": "Beta",
    "graph_mean_degree": "Graph Mean Degree",
    "lambda_2": "Algebraic Connectivity",
    "transitivity": "Transitivity",
    "final_num_lang": "Final Number of Languages",
    "final_num_lang_time": "Time to Reach Final Number of Languages",
    "max_fitness": "Max Fitness",
    "max_self_payoff": "Max Self Payoff",
    "pfix": "Probability of Fixation",
    "fixation_time": "Fixation Time | Fixation",
}

# # %%
# x_item = "num_edge_added"
# legend_item = "num_demes"
# panel_item = "pop_size"

# df = df[df["num_edge_added"] >= 10]

# x_item = "num_demes"
# legend_item = "num_edge_added"

x_item = "lambda_2"
legend_item = "num_demes"

y_item_list = ["pfix", "fixation_time"]

fig, axes = plt.subplots(1, len(y_item_list), figsize=(12 * len(y_item_list), 8), sharex='col')
for i, y_item in enumerate(y_item_list):
    ax = axes[i]
    
    sns.lineplot(data=df_select, x=x_item, y=y_item, hue=legend_item, markers=True, dashes=False, ax=ax, palette="Set1")
    ax.set_title(name_mapping_dict[y_item])
    ax.set_xlabel(f"{name_mapping_dict[x_item]}")
    ax.set_ylabel(name_mapping_dict[y_item])

    ax.axhline(y=df_wm[y_item].item(), color='black', linestyle='--', label='Baseline')
    # ax.plot(df_base_line[x_item], df_base_line[y_item], color='black', linestyle='--', label='Baseline')
    ax.get_legend().remove()

ax.legend(title=name_mapping_dict[legend_item], bbox_to_anchor=(1.05, 1), loc='upper left')
# fig.suptitle(f"{name_mapping_dict["num_edge_added"]}: {num_edge_added_select}")
fig.tight_layout()
# plt.savefig(BASE_PATH / "figures" / f"evolve_x_{x_item}_legend_{legend_item}_pop_{pop_size_select}.png", dpi = 300, bbox_inches='tight')

fig.savefig(
    BASE_PATH / "figures" / f"invade_x_{x_item}_legend_{legend_item}_pop_{100}.png", dpi = 300, bbox_inches='tight',
)
plt.show()

# %%
# fig, axe = plt.subplots(figsize=(8, 6))
# sns.lineplot(data=df, x="deme_size", y="pfix", hue="num_demes", marker='o', errorbar=("sd"), palette="tab10")
# # plt.title(r"Probability of Fixation vs Number of Edges Added ($N_T = 1000$)")
# plt.xlabel("deme size")
# plt.ylabel("Probability of Fixation")
# plt.tight_layout()
# # axe.get_legend().remove()
# # plt.show()
# fig.savefig(
#     BASE_PATH / "figures" / "invade_pfix_vs_deme_size.jpg",
#     dpi=600
# )

# %%
# fig, axe = plt.subplots(figsize=(8, 6))
# sns.lineplot(data=df, x="num_demes", y="fixation_time", hue="deme_size", marker='o', errorbar=("sd"), palette="tab10")
# # plt.title(r"Fixation Time vs Number of Edges Added ($N_T = 1000$)")
# plt.xlabel("Number of demes")
# plt.ylabel("Fixation Time | Fixation")
# plt.tight_layout()
# # axe.get_legend().remove()
# plt.yscale('log')
# # plt.show()
# fig.savefig(
#     BASE_PATH / "figures" / "invade_fixation_time_vs_n_demes.jpg",
#     dpi=600
# )

# %%
# fig, axe = plt.subplots(figsize=(8, 6))
# sns.lineplot(data=df, x="deme_size", y="fixation_time", hue="num_demes", marker='o', errorbar=("sd"), palette="tab10")
# # plt.title(r"Fixation Time vs Number of Edges Added ($N_T = 1000$)")
# plt.xlabel("deme size")
# plt.ylabel("Fixation Time | Fixation")
# plt.tight_layout()
# # axe.get_legend().remove()
# plt.yscale('log')
# # plt.show()
# fig.savefig(
#     BASE_PATH / "figures" / "invade_fixation_time_vs_demesize.jpg",
#     dpi=600
# )

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
