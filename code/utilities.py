import numpy as np

EPS = 1e-6

def load_G(f):
        G = dict()
        el = np.loadtxt(f).astype(int)
        for edge in el:
            n1, n2 = edge
            if n1 in G:
                G[n1] += [n2]
            else:
                G[n1] = [n2]

            if n2 in G:
                G[n2] += [n1]
            else:
                G[n2] = [n1]
        return G
    

def Softmax(x : np.array, temperature : float = 1.0, dim : int = -1) -> np.array:
    x = x / temperature
    x = x - x.min(axis=dim, keepdims=True)
    return np.exp(x) / np.exp(x).sum(axis=dim, keepdims=True)


def Normalize(x : np.array, dim : int = -1) -> np.array:
    return x / (x.sum(axis=dim, keepdims=True) + EPS)


def NormalizeEPS(x : np.array, dim : int = -1) -> np.array:
    return (x + EPS) / (x.sum(axis=dim, keepdims=True) + EPS * x.shape[dim])


def fitness_rank(score, s = 1):
    p = score.argsort().argsort()
    return (1 - s + (2 * s * (p) ) / (len(p) - 1)).clip(min = 0)


def rename_numbers(lst):
    unique_values = sorted(set(lst))  # Get unique values sorted
    mapping = {val: i for i, val in enumerate(unique_values)}  # Create mapping
    return [mapping[val] for val in lst]  # Replace values using mapping
