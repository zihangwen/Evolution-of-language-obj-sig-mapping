import numpy as np
from scipy.optimize import linear_sum_assignment ##### added #####
from sklearn.metrics import confusion_matrix ##### added #####


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
    

def Softmax(x : np.array, dim : int = -1) -> np.array:
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


def align_cluster_labels(y_true, y_pred):
    """
    Aligns the cluster labels of y_pred to match y_true as closely as possible.
    
    Parameters:
        y_true (array-like): Reference clustering labels (length N)
        y_pred (array-like): Clustering labels to be aligned (length N)
        
    Returns:
        y_pred_aligned (numpy array): y_pred with relabeled clusters
    """
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    
    # Build confusion matrix
    contingency_matrix = confusion_matrix(y_true, y_pred)

    # Remove zero rows and columns
    nonzero_rows = contingency_matrix.sum(axis=1) > 0
    nonzero_cols = contingency_matrix.sum(axis=0) > 0
    filtered_matrix = contingency_matrix[np.ix_(nonzero_rows, nonzero_cols)]

    # Solve optimal assignment
    row_ind, col_ind = linear_sum_assignment(-filtered_matrix)
    
    # Create mapping from old labels to new labels
    mapping = {unique_pred[col]: unique_true[row] for row, col in zip(row_ind, col_ind)}
    
    # Apply mapping
    y_pred_aligned = np.array([mapping[label] if label in mapping else label for label in y_pred])
    
    return y_pred_aligned