# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path

BASE_PATH = Path("/home/zihangw/EvoComm")
plt.rcParams.update({'font.size': 16})

# %%
# Load and merge
df_invade = pd.read_csv(BASE_PATH / "results_cpp_combined" / "param_demes_invade.csv", sep="\t")
df_eq     = pd.read_csv(BASE_PATH / "results_cpp_combined" / "param_demes_invade_eq.csv", sep="\t")
df_info   = pd.read_csv(BASE_PATH / "results_cpp_combined" / "param_demes_info.csv", sep="\t")

df_info = df_info.drop(columns="graph_name")  # avoid duplicate column on merge
df_invade = df_invade.merge(df_info, on="graph_path")
df_eq     = df_eq.merge(df_info, on="graph_path")

df_invade["pfix_N"]             = df_invade["pfix"] * df_invade["pop_size"]
df_invade["fixation_time_norm"] = df_invade["fixation_time"] / df_invade["pop_size"]
df_eq["pfix_N"]                 = df_eq["pfix"] * df_eq["pop_size"]
df_eq["fixation_time_norm"]     = df_eq["fixation_time"] / df_eq["pop_size"]

# Network type label
def network_type(path):
    if "wm" in path:             return "wm"
    if "bottleneck_pop" in path: return "fixpop"
    if "demes1_" in path:        return "single_deme"
    return "varpop"

df_invade["network_type"] = df_invade["graph_path"].apply(network_type)
df_eq["network_type"]     = df_eq["graph_path"].apply(network_type)

def for_fixtime(df):
    """Drop rows with no fixation events before plotting fixation_time."""
    return df[df["fixation_count"] > 0]

def wm_fixtime(df_wm):
    """Return wm fixation_time baseline, or None if no fixations occurred."""
    return df_wm["fixation_time_norm"].values[0] if df_wm["fixation_count"].values[0] > 0 else None

# %%
name_map = {
    "algebraic_connectivity": "Algebraic Connectivity",
    "num_edge_added":         "Inter-deme Edges per Pair",
    "num_demes":              "Number of Demes",
    "deme_size":              "Deme Size",
    "pfix_N":                 "pfix × N",
    "fixation_time_norm":     "Fixation Time / N | Fixation",
    "modularity":             "Modularity",
}

(BASE_PATH / "figures").mkdir(exist_ok=True)

# %%
# ---- change this to switch x-axis ----
x_item = "modularity"   # e.g. "algebraic_connectivity", "num_edge_added"
# ---------------------------------------

# %%
# ===== Figure 1: Fixed-pop networks, 4 invades 3 =====
df_fp_inv = df_invade[df_invade["network_type"] == "fixpop"]
df_wm_inv = df_invade[df_invade["network_type"] == "wm"]

fig, axes = plt.subplots(1, 2, figsize=(20, 7))

sns.lineplot(data=df_fp_inv, x=x_item, y="pfix_N",
             hue="num_demes", marker="o", dashes=False, ax=axes[0], palette="Set1")
axes[0].axhline(y=1.0, color="black", linestyle="--", label="neutral")
axes[0].set_xlabel(name_map.get(x_item, x_item))
axes[0].set_ylabel(name_map["pfix_N"])
axes[0].get_legend().remove()

sns.lineplot(data=for_fixtime(df_fp_inv), x=x_item, y="fixation_time_norm",
             hue="num_demes", marker="o", dashes=False, ax=axes[1], palette="Set1")
if wm_fixtime(df_wm_inv) is not None:
    axes[1].axhline(y=wm_fixtime(df_wm_inv), color="black", linestyle="--")
axes[1].set_xlabel(name_map.get(x_item, x_item))
axes[1].set_ylabel(name_map["fixation_time_norm"])
axes[1].set_yscale("log")
axes[1].get_legend().remove()

axes[-1].legend(title=name_map["num_demes"], bbox_to_anchor=(1.05, 1), loc="upper left")
fig.suptitle("Fixed Total Population (N=100): Language 4 Invades Language 3")
fig.tight_layout()
fig.savefig(BASE_PATH / "figures" / f"invade_fixpop_vs_{x_item}.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
# ===== Figure 2: Variable-deme networks, 4 invades 3 =====
df_vp_inv  = df_invade[df_invade["network_type"] == "varpop"]
deme_sizes = sorted(df_vp_inv["deme_size"].unique())

fig, axes = plt.subplots(2, len(deme_sizes), figsize=(10 * len(deme_sizes), 14), sharey="row")
for col, ds in enumerate(deme_sizes):
    df_sub = df_vp_inv[df_vp_inv["deme_size"] == ds]
    for row, y_item in enumerate(["pfix_N", "fixation_time_norm"]):
        ax = axes[row][col]
        data = for_fixtime(df_sub) if y_item == "fixation_time_norm" else df_sub
        sns.lineplot(data=data, x=x_item, y=y_item,
                     hue="num_demes", marker="o", dashes=False, ax=ax, palette="Set1")
        ax.set_xlabel(name_map.get(x_item, x_item) if row == 1 else "")
        ax.set_ylabel(name_map[y_item] if col == 0 else "")
        if row == 0:
            ax.set_title(f"Deme Size = {ds}")
        if row == 1:
            ax.set_yscale("log")
        ax.get_legend().remove()

axes[0][-1].legend(title=name_map["num_demes"], bbox_to_anchor=(1.05, 1), loc="upper left")
fig.suptitle("Variable Deme Structure: Language 4 Invades Language 3")
fig.tight_layout()
fig.savefig(BASE_PATH / "figures" / f"invade_varpop_vs_{x_item}.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
# ===== Figure 3: Fixed-pop, equal payoff (4 vs 4) =====
df_fp_eq = df_eq[df_eq["network_type"] == "fixpop"]
df_wm_eq = df_eq[df_eq["network_type"] == "wm"]

fig, axes = plt.subplots(1, 2, figsize=(20, 7))
for ax, y_item in zip(axes, ["pfix_N", "fixation_time_norm"]):
    data = for_fixtime(df_fp_eq) if y_item == "fixation_time_norm" else df_fp_eq
    sns.lineplot(data=data, x=x_item, y=y_item,
                 hue="num_demes", marker="o", dashes=False, ax=ax, palette="Set1")
    wm_val = 1.0 if y_item == "pfix_N" else wm_fixtime(df_wm_eq)
    if wm_val is not None:
        ax.axhline(y=wm_val, color="black", linestyle="--", label="wm baseline")
    ax.set_xlabel(name_map.get(x_item, x_item))
    ax.set_ylabel(name_map[y_item])
    if y_item == "fixation_time_norm":
        ax.set_yscale("log")
    ax.get_legend().remove()

axes[-1].legend(title=name_map["num_demes"], bbox_to_anchor=(1.05, 1), loc="upper left")
fig.suptitle("Fixed Total Population (N=100): Equal Payoff (Language 4 vs 4)")
fig.tight_layout()
fig.savefig(BASE_PATH / "figures" / f"eq_fixpop_vs_{x_item}.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
# ===== Figure 4: Variable-deme, equal payoff (4 vs 4) =====
df_vp_eq   = df_eq[df_eq["network_type"] == "varpop"]
deme_sizes = sorted(df_vp_eq["deme_size"].unique())

fig, axes = plt.subplots(2, len(deme_sizes), figsize=(10 * len(deme_sizes), 14), sharey="row")
for col, ds in enumerate(deme_sizes):
    df_sub = df_vp_eq[df_vp_eq["deme_size"] == ds]
    for row, y_item in enumerate(["pfix_N", "fixation_time_norm"]):
        ax = axes[row][col]
        data = for_fixtime(df_sub) if y_item == "fixation_time_norm" else df_sub
        sns.lineplot(data=data, x=x_item, y=y_item,
                     hue="num_demes", marker="o", dashes=False, ax=ax, palette="Set1")
        ax.set_xlabel(name_map.get(x_item, x_item) if row == 1 else "")
        ax.set_ylabel(name_map[y_item] if col == 0 else "")
        if row == 0:
            ax.set_title(f"Deme Size = {ds}")
        if row == 1:
            ax.set_yscale("log")
        ax.get_legend().remove()

axes[0][-1].legend(title=name_map["num_demes"], bbox_to_anchor=(1.05, 1), loc="upper left")
fig.suptitle("Variable Deme Structure: Equal Payoff (Language 4 vs 4)")
fig.tight_layout()
fig.savefig(BASE_PATH / "figures" / f"eq_varpop_vs_{x_item}.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
# ===== Figure 5: Side-by-side comparison, fixed-pop =====
fig, axes = plt.subplots(1, 4, figsize=(36, 7))

plot_specs = [
    (df_fp_inv, "pfix_N",             "4 invades 3: pfix×N",           1.0),
    (df_fp_inv, "fixation_time_norm", "4 invades 3: fixation time",    wm_fixtime(df_wm_inv)),
    (df_fp_eq,  "pfix_N",             "Equal (4 vs 4): pfix×N",        1.0),
    (df_fp_eq,  "fixation_time_norm", "Equal (4 vs 4): fixation time", wm_fixtime(df_wm_eq)),
]

for ax, (df_sub, y_item, title, baseline) in zip(axes, plot_specs):
    data = for_fixtime(df_sub) if y_item == "fixation_time_norm" else df_sub
    sns.lineplot(data=data, x=x_item, y=y_item,
                 hue="num_demes", marker="o", dashes=False, ax=ax, palette="Set1")
    if baseline is not None:
        ax.axhline(y=baseline, color="black", linestyle="--", label="wm baseline")
    ax.set_title(title)
    ax.set_xlabel(name_map.get(x_item, x_item))
    ax.set_ylabel("")
    if "time" in y_item:
        ax.set_yscale("log")
    ax.get_legend().remove()

axes[-1].legend(title=name_map["num_demes"], bbox_to_anchor=(1.05, 1), loc="upper left")
fig.suptitle("Fixed Total Population (N=100): Invasion vs Equal Payoff")
fig.tight_layout()
fig.savefig(BASE_PATH / "figures" / f"compare_fixpop_invasion_vs_eq_{x_item}.png", dpi=300, bbox_inches="tight")
plt.show()

# %%