import networkx as nx

def generate(n=1000, k=10, beta=0.08, communities=6):
    """
    Industrial/organizational era:
    - Watts–Strogatz per community; sparse inter-community links
    """
    per = max(10, n // communities)
    blocks = []
    offset = 0
    G = nx.Graph()
    for _ in range(communities):
        H = nx.watts_strogatz_graph(per, k, beta)
        mapping = {i: i + offset for i in range(per)}
        H = nx.relabel_nodes(H, mapping)
        G = nx.compose(G, H)
        blocks.append(list(mapping.values()))
        offset += per
    # sparse inter-community chain
    for i in range(communities - 1):
        if blocks[i] and blocks[i+1]:
            G.add_edge(blocks[i][0], blocks[i+1][0])
    return G
