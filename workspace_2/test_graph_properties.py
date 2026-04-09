# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path

BASE_PATH = Path("/home/zihangw/EvoComm")
plt.rcParams.update({'font.size': 14})

# %%
# Load and merge
df_invade = pd.read_csv(BASE_PATH / "results_cpp_combined" / "param_demes_invade.csv", sep="\t")
df_eq     = pd.read_csv(BASE_PATH / "results_cpp_combined" / "param_demes_invade_eq.csv", sep="\t")
df_info   = pd.read_csv(BASE_PATH / "results_cpp_combined" / "param_demes_info.csv", sep="\t")

df_info = df_info.drop(columns="graph_name")
df_invade = df_invade.merge(df_info, on="graph_path")
df_eq     = df_eq.merge(df_info, on="graph_path")

df_invade["pfix_N"]             = df_invade["pfix"] * df_invade["pop_size"]
df_invade["fixation_time_norm"] = df_invade["fixation_time"] / df_invade["pop_size"]
df_eq["pfix_N"]                 = df_eq["pfix"] * df_eq["pop_size"]
df_eq["fixation_time_norm"]     = df_eq["fixation_time"] / df_eq["pop_size"]

def network_type(path):
    if "wm" in path:             return "wm"
    if "bottleneck_pop" in path: return "fixpop"
    if "demes1_" in path:        return "single_deme"
    return "varpop"

df_invade["network_type"] = df_invade["graph_path"].apply(network_type)
df_eq["network_type"]     = df_eq["graph_path"].apply(network_type)

(BASE_PATH / "figures").mkdir(exist_ok=True)

# %%
graph_props = [
    "algebraic_connectivity",
    "modularity",
    "graph_mean_degree",
    "average_clustering",
    "transitivity",
    "diameter",
    "assortativity",
    "num_edges",
    "num_demes",
    "deme_size",
    "num_edge_added",
]

name_map = {
    "algebraic_connectivity": "Algebraic Connectivity",
    "modularity":             "Modularity",
    "graph_mean_degree":      "Mean Degree",
    "average_clustering":     "Avg Clustering",
    "transitivity":           "Transitivity",
    "diameter":               "Diameter",
    "assortativity":          "Assortativity",
    "num_edges":              "Num Edges",
    "num_demes":              "Num Demes",
    "deme_size":              "Deme Size",
    "num_edge_added":         "Inter-deme Edges / Pair",
    "pfix_N":                 "pfix × N",
    "fixation_time_norm":     "Fixation Time / N",
}

def corr_matrix(df):
    """Compute Pearson r between each graph property and each outcome.
    Uses fixation_count > 0 filter only for fixation_time_norm."""
    rows = []
    for prop in graph_props:
        r_pfix = df[[prop, "pfix_N"]].dropna().corr().iloc[0, 1]
        df_ft  = df[df["fixation_count"] > 0]
        r_ft   = df_ft[[prop, "fixation_time_norm"]].dropna().corr().iloc[0, 1]
        rows.append({"prop": prop, "pfix_N": r_pfix, "fixation_time_norm": r_ft})
    mat = pd.DataFrame(rows).set_index("prop")
    mat.index   = [name_map[p] for p in graph_props]
    mat.columns = [name_map[c] for c in mat.columns]
    return mat

# %%
# 4 subsets: (invade / eq) × (fixpop / varpop)
subsets = {
    "4 invades 3\n(fixpop)":   df_invade[df_invade["network_type"] == "fixpop"],
    "4 invades 3\n(varpop)":   df_invade[df_invade["network_type"] == "varpop"],
    "Equal payoff\n(fixpop)":  df_eq[df_eq["network_type"] == "fixpop"],
    "Equal payoff\n(varpop)":  df_eq[df_eq["network_type"] == "varpop"],
}

# %%
# ===== Figure 1: 2×2 grid of correlation heatmaps =====
fig, axes = plt.subplots(2, 2, figsize=(14, 22))

for ax, (title, df_sub) in zip(axes.flat, subsets.items()):
    mat = corr_matrix(df_sub)
    sns.heatmap(mat.astype(float), annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                linewidths=0.5, ax=ax, cbar=True)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

fig.suptitle("Pearson r: Graph Properties vs Outcomes", fontsize=16, fontweight="bold")
fig.tight_layout()
fig.savefig(BASE_PATH / "figures" / "prop_corr_heatmap_2x2.png", dpi=200, bbox_inches="tight")
plt.show()

# %%
# ===== Figure 2 & 3: Scatter grids — 4 rows (subsets) × 11 cols (properties) =====
# One figure per outcome variable
subset_titles = list(subsets.keys())
n_props = len(graph_props)
n_rows  = len(subsets)

for outcome in ["pfix_N", "fixation_time_norm"]:
    fig, axes = plt.subplots(n_rows, n_props,
                             figsize=(4 * n_props, 4 * n_rows),
                             squeeze=False)

    for row, (title, df_sub) in enumerate(subsets.items()):
        data = df_sub[df_sub["fixation_count"] > 0] if outcome == "fixation_time_norm" else df_sub

        for col, prop in enumerate(graph_props):
            ax = axes[row][col]
            cols = list(dict.fromkeys([prop, outcome, "num_demes"]))
            df_cell = data[cols].dropna()

            sns.scatterplot(data=df_cell, x=prop, y=outcome, hue="num_demes",
                            palette="Set1", s=30, alpha=0.8, ax=ax,
                            legend=(row == 0 and col == n_props - 1))

            r = df_cell[[prop, outcome]].corr().iloc[0, 1]
            ax.set_title(f"r = {r:.2f}", fontsize=10)
            ax.set_xlabel(name_map.get(prop, prop) if row == n_rows - 1 else "")
            ax.set_ylabel(title.replace("\n", " ") if col == 0 else "")
            if outcome == "fixation_time_norm":
                ax.set_yscale("log")

    # shared legend in top-right panel
    axes[0][-1].legend(title=name_map["num_demes"], bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)

    fig.suptitle(f"Scatter: Graph Properties vs {name_map[outcome]}", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(BASE_PATH / "figures" / f"prop_scatter_{outcome}.png", dpi=300, bbox_inches="tight")
    plt.show()

# %%
# ===== Figure 2: Pairwise property correlation (sanity check on multicollinearity) =====
df_all = pd.concat([
    df_invade[df_invade["network_type"].isin(["fixpop", "varpop"])],
    df_eq[df_eq["network_type"].isin(["fixpop", "varpop"])],
]).drop_duplicates(subset="graph_path")

corr_props = df_all[graph_props].corr()
corr_props.index   = [name_map[p] for p in graph_props]
corr_props.columns = [name_map[p] for p in graph_props]

fig, ax = plt.subplots(figsize=(11, 10))
sns.heatmap(corr_props, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            linewidths=0.5, ax=ax)
ax.set_title("Pairwise Correlation Between Graph Properties", fontsize=14, fontweight="bold")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
fig.tight_layout()
fig.savefig(BASE_PATH / "figures" / "prop_heatmap_pairwise.png", dpi=200, bbox_inches="tight")
plt.show()

# %%
