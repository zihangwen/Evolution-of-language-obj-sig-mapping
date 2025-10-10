import networkx as nx
import numpy as np
import pandas as pd

def _safe_avg_path_length(G):
    U = G.to_undirected()
    if nx.is_connected(U):
        return nx.average_shortest_path_length(U)
    comp = max(nx.connected_components(U), key=len)
    return nx.average_shortest_path_length(U.subgraph(comp).copy())

def _degree_tail_fit(deg_seq):
    s = np.array([d for d in deg_seq if d > 0], dtype=float)
    if len(s) == 0:
        return {"lognorm_mu": float("nan"), "lognorm_sigma": float("nan")}
    mu = float(np.mean(np.log(s)))
    sigma = float(np.std(np.log(s)))
    return {"lognorm_mu": mu, "lognorm_sigma": sigma}

def summarize_graph(G, label=None, attributes=None):
    deg_values = [d for _, d in G.degree()]
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G)
    clustering = nx.average_clustering(G.to_undirected())
    try:
        assort = nx.degree_pearson_correlation_coefficient(G.to_undirected())
    except Exception:
        assort = float("nan")
    apl = _safe_avg_path_length(G)
    U = G.to_undirected()
    cc_sizes = [len(c) for c in nx.connected_components(U)]
    giant_frac = max(cc_sizes)/n if cc_sizes else 0.0
    tail = _degree_tail_fit(deg_values)
    res = {
        "label": label or "",
        "n": n, "m": m, "density": density,
        "avg_degree": float(np.mean(deg_values) if deg_values else 0.0),
        "deg_var": float(np.var(deg_values) if deg_values else 0.0),
        "clustering": clustering,
        "assort_degree": assort,
        "avg_path_len_lcc": apl,
        "giant_component_frac": giant_frac,
        "deg_lognorm_mu": tail["lognorm_mu"],
        "deg_lognorm_sigma": tail["lognorm_sigma"],
    }
    if attributes and isinstance(attributes, dict):
        res.update(attributes)
    return res

def write_summary_csv(graphs, out_csv):
    rows = []
    for (label, G, attrs) in graphs:
        rows.append(summarize_graph(G, label=label, attributes=attrs))
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df
