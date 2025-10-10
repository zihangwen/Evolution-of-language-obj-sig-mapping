import networkx as nx
import numpy as np

def generate(n=800, blocks=(0.35,0.30,0.20,0.15), p_within=(0.08,0.06,0.05,0.04), p_between=0.003, assort_boost=1.0, seed=None):
    """
    Agrarian village stylized generator (SBM-like):
    - Strong assortativity by caste/lineage (blocks)
    - Sparse bridges across blocks
    """
    rng = np.random.default_rng(seed)
    sizes = (np.array(blocks)/np.sum(blocks) * n).astype(int)
    B = len(sizes)
    P = np.full((B,B), p_between, dtype=float)
    for i, pin in enumerate(p_within[:B]):
        P[i,i] = pin * assort_boost
    G = nx.stochastic_block_model(sizes.tolist(), P, seed=seed)
    return G
