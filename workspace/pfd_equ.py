# %%
import numpy as np
import einops
import matplotlib.pyplot as plt
import random
# import math
# from collections import defaultdict
import networkx as nx

EPS = 1e-6

# %%
def NormalizeEPS(x : np.array, dim : int = -1) -> np.array:
    result = (x + EPS) / (x.sum(axis=dim, keepdims=True) + EPS * x.shape[dim])
    return np.round(result, 4)

# %%
P1 = np.array([[1., 0., 0., 0., 0.],
               [0., 1., 0., 0., 0.],
               [1., 0., 0., 0., 0.],
               [0., 1., 0., 0., 0.],
               [0., 0., 0., 1., 0.]])

Q1 = NormalizeEPS(P1.T)

# %%
P2 = np.array([[0., 0., 1., 0., 0.],
               [0., 0., 0., 1., 0.],
               [1., 0., 0., 0., 0.],
               [0., 1., 0., 0., 0.],
               [1., 0., 0., 0., 0.]])

Q2 = NormalizeEPS(P2.T)

# %%
c11 = einops.einsum(P1, Q1, 'obj sig, sig obj ->')
c12 = einops.einsum(P1, Q2, 'obj sig, sig obj ->')
c21 = einops.einsum(P2, Q1, 'obj sig, sig obj ->')
c22 = einops.einsum(P2, Q2, 'obj sig, sig obj ->')

# %%
propotion_2 = np.linspace(0, 1, 100)
average_payoff_1 = lambda prop_2: (1 - prop_2) * c11 + prop_2 * 1/2 * (c12 + c21)
average_payoff_2 = lambda prop_2: prop_2 * c22 + (1 - prop_2) * 1/2 * (c12 + c21)

plt.figure(figsize=(8, 6))
plt.plot(propotion_2, average_payoff_1(propotion_2), label='Language 1 Speaker', color='blue')
plt.plot(propotion_2, average_payoff_2(propotion_2), label='Language 2 Speaker', color='orange')
plt.title('Speaker Payoff vs Proportion of Language 2 Speakers in Neighbors')
plt.xlabel('Proportion of Language 2 Speakers in Neighbors')
plt.ylabel('Payoff')
plt.legend()
plt.grid()
plt.show()

# %%
# a type is language 1 speaker, A type is language 2 speaker
def neighbor_A_fraction(G: nx.Graph, is_A: dict):
    """
    is_A: dict {node: bool} indicating whether node is type A (True) or a (False)
    Returns: dict {node: frac_of_neighbors_that_are_A} (NaN for isolated nodes)
    """
    out = {}
    for v in G.nodes:
        nbrs = list(G.neighbors(v))
        d = len(nbrs)
        if d == 0:
            out[v] = float("nan")
            continue
        out[v] = sum(is_A[u] for u in nbrs) / d
    return out

def sample_type_level_stats(G: nx.Graph, ks=range(0, 101), samples_per_k=2000, seed=42):
    rng = random.Random(seed)
    nodes = list(G.nodes)

    out = {}  # out[k] = {"A_mean":..., "a_mean":...}
    for k in ks:
        # A_fracs = []
        # a_fracs = []
        a_payoff_list = []
        A_payoff_list = []
        for _ in range(samples_per_k):
            A_set = set(rng.sample(nodes, k))
            is_A = {v: (v in A_set) for v in nodes}
            freq = neighbor_A_fraction(G, is_A)

            A_vals = [freq[v] for v in nodes if is_A[v]]
            a_vals = [freq[v] for v in nodes if (not is_A[v])]

            a_payoff_sum = average_payoff_1(np.array(a_vals)).sum()
            A_payoff_sum = average_payoff_2(np.array(A_vals)).sum()
            
            A_payoff_list.append(A_payoff_sum)
            a_payoff_list.append(a_payoff_sum)

        out[k] = {
            "A_payoff_samples": A_payoff_list,
            "a_payoff_samples": a_payoff_list,
        }
            # A_fracs.append(sum(A_vals))
            # a_fracs.append(sum(a_vals))

        # out[k] = {
        #     "A_neighbor_A_frac": A_fracs,
        #     "a_neighbor_A_frac": a_fracs,
        # }
    return out

# %%
graph_path = "/home/zihangw/EvoComm/networks/bottleneck_pop100/bn_ndeme5_edge10_beta0_rep0.txt"
G = nx.read_edgelist(graph_path, nodetype=int)

test_stats = sample_type_level_stats(G, ks=range(0, 101), samples_per_k=5000, seed=42)

# %%
plt.figure(figsize=(8, 6))
A_means = [np.mean(test_stats[k]["A_payoff_samples"]) for k in range(0, 101)]
a_means = [np.mean(test_stats[k]["a_payoff_samples"]) for k in range(0, 101)]
plt.plot(range(0, 101), a_means, label='Language 1 Speaker', color='blue')
plt.plot(range(0, 101), A_means, label='Language 2 Speaker', color='orange')
# std
a_stds = [np.std(test_stats[k]["a_payoff_samples"]) for k in range(0, 101)]
A_stds = [np.std(test_stats[k]["A_payoff_samples"]) for k in range(0, 101)]
plt.fill_between(range(0, 101), np.array(a_means) - np.array(a_stds), np.array(a_means) + np.array(a_stds), color='blue', alpha=0.5)
plt.fill_between(range(0, 101), np.array(A_means) - np.array(A_stds), np.array(A_means) + np.array(A_stds), color='orange', alpha=0.5)
plt.title('Speaker (language) Payoff vs Number of Language 2 Speakers in Population')
plt.xlabel('Number of Language 2 Speakers in Population')
plt.ylabel('Total Payoff')
plt.legend()
plt.grid()
plt.show()

# %%
