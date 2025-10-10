import networkx as nx
import numpy as np

def generate(n_families=30, p_marriage=0.06, p_business=0.08, interlayer_coupling=0.3, seed=None):
    """
    Early-modern elites (multiplex proxy):
    - Layer 1: marriage alliances between families
    - Layer 2: business ties between families
    - Interlayer coupling increases probability of dual ties (overlap)
    Returns a simple aggregated graph (union of layers).
    """
    rng = np.random.default_rng(seed)
    Gm = nx.erdos_renyi_graph(n_families, p_marriage, seed=seed)
    Gb = nx.erdos_renyi_graph(n_families, p_business, seed=seed)
    for u, v in list(Gm.edges()):
        if not Gb.has_edge(u, v) and rng.random() < interlayer_coupling:
            Gb.add_edge(u, v)
    for u, v in list(Gb.edges()):
        if not Gm.has_edge(u, v) and rng.random() < interlayer_coupling:
            Gm.add_edge(u, v)
    G = nx.Graph()
    G.add_nodes_from(range(n_families))
    G.add_edges_from(Gm.edges())
    G.add_edges_from(Gb.edges())
    return G
