import networkx as nx
import numpy as np

def generate(num_camps=6, mean_camp_size=30, camp_size_sd=5, p_intra=0.15, p_inter=0.01, household_clique_p=0.6, seed=None):
    """
    Hunter-gatherer stylized generator:
    - Camps as blocks (SBM-like) with high clustering
    - Occasional inter-camp visits (weak bridges)
    - Optional household cliques to boost transitivity
    """
    rng = np.random.default_rng(seed)
    sizes = np.maximum(5, rng.normal(mean_camp_size, camp_size_sd, num_camps).astype(int))
    P = np.full((num_camps, num_camps), p_inter, dtype=float)
    np.fill_diagonal(P, p_intra)
    G = nx.stochastic_block_model(sizes.tolist(), P, seed=seed)
    # add household cliques within each block
    start = 0
    for sz in sizes.tolist():
        nodes = list(range(start, start+sz))
        start += sz
        rng.shuffle(nodes)
        for i in range(0, len(nodes), 4):
            hh = nodes[i:i+4]
            if len(hh) > 1 and rng.random() < household_clique_p:
                for u in hh:
                    for v in hh:
                        if u < v:
                            G.add_edge(u, v)
    return G
