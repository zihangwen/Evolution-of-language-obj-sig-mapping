import networkx as nx
import numpy as np

def generate(n_cities=200, p_backbone=0.015, geo_radius=0.2, gravity_exponent=1.0, seed=None):
    """
    Imperial/medieval trade backbone (stylized):
    - Cities uniformly scattered on unit square
    - Base random geometric proximity graph
    - Extra long-range edges by gravity (mass * mass / distance^alpha)
    """
    rng = np.random.default_rng(seed)
    pos = {i: rng.random(2) for i in range(n_cities)}
    G = nx.random_geometric_graph(n_cities, geo_radius, pos=pos, seed=seed)
    for i in G.nodes():
        G.nodes[i]["mass"] = float(np.exp(rng.normal(0, 1.0)))
    nodes = list(G.nodes())
    num_trials = int(p_backbone * n_cities * n_cities * 0.5)
    for _ in range(num_trials):
        u, v = rng.choice(nodes, 2, replace=False)
        duv = np.linalg.norm(pos[u] - pos[v]) + 1e-6
        puv = (G.nodes[u]["mass"] * G.nodes[v]["mass"]) / (duv ** gravity_exponent)
        if rng.random() < puv * 1e-3:
            G.add_edge(u, v)
    return G
