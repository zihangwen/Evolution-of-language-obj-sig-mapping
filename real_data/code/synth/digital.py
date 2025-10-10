import networkx as nx
import numpy as np

def generate(n=2000, m=3, triadic_closure=0.3, seed=None):
    """
    Contemporary digital-like:
    - Preferential attachment backbone (heavy tails)
    - Triadic closure to increase clustering
    """
    rng = np.random.default_rng(seed)
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    for u in range(n):
        nbrs = list(G.neighbors(u))
        ln = len(nbrs)
        for i in range(ln):
            for j in range(i+1, ln):
                a, b = nbrs[i], nbrs[j]
                if not G.has_edge(a, b) and rng.random() < triadic_closure:
                    G.add_edge(a, b)
    return G
