import networkx as nx
import torch
from random import randint
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import numpy as np

edge_colors = {
    "inv": "green",
    "spu": "blue",
    "added": "red"
}
node_colors = {
    True: "red",
    False: "#1f78b4"
}


def remove_from_graph(G, what_to_remove):
    G = G.copy()
    edge_remove = []
    for (u,v), val in nx.get_edge_attributes(G, 'origin').items():
        if val == what_to_remove:
            edge_remove.append((u,v))
    G.remove_edges_from(edge_remove)
    G.remove_edges_from([(v,u) for v,u in G.edges() if not G.has_edge(u,v)])
    G.remove_nodes_from(list(nx.isolates(G)))
    return G

def mark_edges(G, inv_edge_index, spu_edge_index):
    nx.set_edge_attributes(
        G,
        name="origin",
        values={(u.item(), v.item()): "inv" for u,v in inv_edge_index.T}
    )
    nx.set_edge_attributes(
        G,
        name="origin",
        values={(u.item(), v.item()): "spu" for u,v in spu_edge_index.T}
    )

def mark_frontier(G, G_filt):
    # mark frontier nodes as nodes attached to both inv and spu parts
    # to mark nodes check which nodes have a change in the degree between original and filtered graph
    frontier = []
    for n in G_filt.nodes():
        if G.degree[n] != G_filt.degree[n]:                    
            frontier.append(n)            
    nx.set_node_attributes(G_filt, name="frontier", values=False)
    nx.set_node_attributes(G_filt, name="frontier", values={n: True for n in frontier})
    return len(frontier)

def draw(G, name, pos=None):
    if pos is None:
        pos = nx.kamada_kawai_layout(G)
    nx.draw(
        G,
        with_labels = True,
        pos=pos,
        edge_color=list(map(lambda x: edge_colors[x], nx.get_edge_attributes(G,'origin').values())),
        node_color=list(map(lambda x: node_colors[x], [nx.get_node_attributes(G,'frontier').get(n, False) for n in G.nodes()])),
    )
    plt.savefig(f'GOOD/kernel/pipelines/plots/{name}.png')
    plt.close()
    return pos

def random_attach(S, T):
    # random attach frontier nodes in S and T

    S_frontier = list(filter(lambda x: nx.get_node_attributes(S,'frontier').get(x, False), S.nodes()))
    T_frontier = list(filter(lambda x: nx.get_node_attributes(T,'frontier').get(x, False), T.nodes()))

    ret = nx.union(S, T, rename=("", "T"))
    for n in S_frontier:
        # pick random node v in G_t_spu
        # add edge (u,v) and (v,u)
        idx = randint(0, len(T_frontier)-1)
        v = "T" + str(T_frontier[idx])

        assert str(n) in ret.nodes() and v in ret.nodes()

        ret.add_edge(str(n), v, origin="added")
        ret.add_edge(v, str(n), origin="added")
    return ret

def expl_acc(expl, data):
    edge_gt = {(u.item(),v.item()): data.edge_gt[i] for i, (u,v) in enumerate(data.edge_index.T)} 
    edge_expl = set([(u.item(),v.item()) for u,v in expl.T])

    
    tp = int(sum([edge_gt[(u.item(),v.item())] for u,v in expl.T]))
    fp = int(sum([not edge_gt[(u.item(),v.item())] for u,v in expl.T]))
    tn = int(sum([not (u.item(),v.item()) in edge_expl and not edge_gt[(u.item(),v.item())] for u,v in data.edge_index.T]))
    fn = int(sum([not (u.item(),v.item()) in edge_expl and edge_gt[(u.item(),v.item())] for u,v in data.edge_index.T]))
    
    acc = (tp + tn) / (tp + fp + tn + fn)
    f1 = 2*tp / (2*tp + fp + fn)
    assert (tp + fp + tn + fn) == len(edge_gt)
    return round(f1, 3)

def sample_edges(G, where_to_sample, alpha_2):
    # keep each spu edge with probability alpha_2
    G = G.copy()
    edge_remove = []
    for (u,v), val in nx.get_edge_attributes(G, 'origin').items():
        if val == where_to_sample:
            if np.random.binomial(1, alpha_2, 1)[0] == 0:
                edge_remove.append((u,v))
    G.remove_edges_from(edge_remove)
    G.remove_edges_from([(v,u) for v,u in G.edges() if not G.has_edge(u,v)])
    G.remove_nodes_from(list(nx.isolates(G)))
    return G