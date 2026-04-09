# %%
import math
import pandas as pd
import os
from pathlib import Path
import networkx as nx

BASE_PATH = Path("/home/zihangw/EvoComm")

param_file        = BASE_PATH / "param_space" / "invade_param_eq.txt"
out_path_base     = BASE_PATH / "results_invade_cpp"
combined_results_path = BASE_PATH / "results_cpp_combined" / "param_demes_invade.csv"
graph_info_path   = BASE_PATH / "results_cpp_combined" / "param_demes_info.csv"

with open(param_file, "r") as f:
    param_sim = [x.strip() for x in f if x.strip()]
param_sim = [x.split(" ") for x in param_sim]


# %%
def classify_network(graph_base, graph_name):
    """
    Extract semantic network properties from folder and file names.
    Returns a dict with: network_type, num_demes, deme_size, inter_deme_edges, rep.

    Supported conventions:
      wm_100/wm_100                            → well-mixed
      bottleneck_pop100/bn_ndemeN_edgeE_...    → fixed total pop, variable deme structure
      bottleneck_demesN_sizeS/bn_N_E_m0_0      → variable demes and deme sizes
    """
    folder = os.path.basename(graph_base)

    if folder.startswith("wm"):
        pop_size = int(folder.split("_")[1])
        return dict(demeconn_number=0, num_demes=1,
                    deme_size=pop_size, num_edge_added=0, rep=0)

    elif folder.startswith("bottleneck_pop"):
        # folder: bottleneck_pop100
        # file:   bn_ndeme5_edge1_beta0_rep0
        pop_size  = int(folder.split("pop")[1])
        num_demes = int(graph_name.split("ndeme")[1].split("_")[0])
        edges     = int(graph_name.split("edge")[1].split("_")[0])
        rep       = int(graph_name.split("rep")[1])
        return dict(demeconn_number=2, num_demes=num_demes,
                    deme_size=pop_size // num_demes, num_edge_added=edges, rep=rep)

    elif folder.startswith("bottleneck_demes"):
        # folder: bottleneck_demes5_size10
        # file:   bn_5_20_m0_0
        parts     = folder.split("_")          # ['bottleneck', 'demes5', 'size10']
        num_demes = int(parts[1].replace("demes", ""))
        deme_size = int(parts[2].replace("size", ""))
        np        = graph_name.split("_")      # ['bn', '5', '20', 'm0', '0']
        edges     = int(np[2])
        rep       = int(np[4])
        demeconn  = 0 if num_demes == 1 else 2
        return dict(demeconn_number=demeconn, num_demes=num_demes,
                    deme_size=deme_size, num_edge_added=edges, rep=rep)

    else:
        return dict(demeconn_number=None, num_demes=None,
                    deme_size=None, num_edge_added=None, rep=None)


# %%
# Part 1: Combine simulation results
df_all = pd.DataFrame()

for param in param_sim:
    graph_path = param[2]
    graph_base = os.path.dirname(graph_path)
    graph_name = os.path.basename(graph_path).split(".")[0]
    out_path   = out_path_base / graph_base / graph_name

    if not out_path.exists():
        print(f"Missing: {out_path}")
        continue

    files = [f for f in os.listdir(out_path) if f.endswith(".txt")]
    dfs   = []
    for f in files:
        df_f = pd.read_csv(out_path / f, sep="\t")
        df_f.columns = [c.strip("# :") for c in df_f.columns]   # '# graph_name:' -> 'graph_name'
        dfs.append(df_f)
    df = pd.concat(dfs, ignore_index=True)

    df["fixation_time_ws"] = df["fixation_time"] * df["fixation_count"]

    agg = df.groupby("graph_name").agg(
        num_trials          =("num_trials",          "sum"),
        fixation_count      =("fixation_count",      "sum"),
        fixation_time_ws    =("fixation_time_ws",    "sum"),
        co_existence_count  =("co_existence_count",  "sum"),
    ).reset_index()

    agg["fixation_time"] = agg.apply(
        lambda r: r["fixation_time_ws"] / r["fixation_count"] if r["fixation_count"] > 0 else 0,
        axis=1
    )
    agg["pfix"]      = agg["fixation_count"]     / agg["num_trials"]
    agg["pco_exist"] = agg["co_existence_count"] / agg["num_trials"]
    agg["graph_path"] = graph_path
    agg = agg.drop(columns="fixation_time_ws")

    df_all = pd.concat([df_all, agg], ignore_index=True)

combined_results_path.parent.mkdir(parents=True, exist_ok=True)
df_all.to_csv(combined_results_path, index=False, sep="\t")
print(f"Saved {len(df_all)} rows -> {combined_results_path}")


# %%
# Part 2: Graph structural info
df_graph = pd.DataFrame()

for param in param_sim:
    graph_path = param[2]
    graph_base = os.path.dirname(graph_path)
    graph_name = os.path.basename(graph_path).split(".")[0]

    net_props = classify_network(graph_base, graph_name)

    G = nx.read_edgelist(BASE_PATH / graph_path, nodetype=int)
    pop_size   = G.number_of_nodes()
    mean_deg   = sum(dict(G.degree()).values()) / pop_size

    def safe(fn):
        try:
            v = fn()
            return None if (v is None or (isinstance(v, float) and math.isnan(v))) else v
        except:
            return None

    row = {
        "graph_path":            graph_path,
        "graph_name":            graph_name,
        "pop_size":              pop_size,
        **net_props,
        "num_nodes":             pop_size,
        "num_edges":             G.number_of_edges(),
        "graph_mean_degree":     mean_deg,
        "average_clustering":    safe(lambda: nx.average_clustering(G)),
        "transitivity":          safe(lambda: nx.transitivity(G)),
        "diameter":              safe(lambda: nx.diameter(G)),
        "modularity":            safe(lambda: nx.community.quality.modularity(G, nx.community.louvain_communities(G))),
        "assortativity":         safe(lambda: nx.degree_assortativity_coefficient(G)),
        "algebraic_connectivity":safe(lambda: nx.algebraic_connectivity(G)),
    }
    df_graph = pd.concat([df_graph, pd.DataFrame([row])], ignore_index=True)

graph_info_path.parent.mkdir(parents=True, exist_ok=True)
df_graph.to_csv(graph_info_path, index=False, sep="\t")
print(f"Saved {len(df_graph)} rows -> {graph_info_path}")

# %%