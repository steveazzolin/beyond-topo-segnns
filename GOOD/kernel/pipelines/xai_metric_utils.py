import networkx as nx
import torch
from random import randint
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import numpy as np
import os

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

def mark_edges(G, inv_edge_index, spu_edge_index, inv_edge_w=None, spu_edge_w=None):
    nx.set_edge_attributes(
        G,
        name="origin",
        values={(u.item(), v.item()): "inv" for u,v in inv_edge_index.T}
    )
    if not inv_edge_w is None:
        nx.set_edge_attributes(
            G,
            name="attn_weight",
            values={(u.item(), v.item()): round(inv_edge_w[i].item(),2) for i, (u,v) in enumerate(inv_edge_index.T)}
        )
    nx.set_edge_attributes(
        G,
        name="origin",
        values={(u.item(), v.item()): "spu" for u,v in spu_edge_index.T}
    )
    if not spu_edge_w is None:
        nx.set_edge_attributes(
            G,
            name="attn_weight",
            values={(u.item(), v.item()): round(spu_edge_w[i].item(),2) for i, (u,v) in enumerate(spu_edge_index.T)}
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

def draw(config, G, name, subfolder="", pos=None):
    plt.figure()
    if pos is None:
        pos = nx.kamada_kawai_layout(G)
    edge_color = list(map(lambda x: edge_colors[x], nx.get_edge_attributes(G,'origin').values()))
    nx.draw(
        G,
        with_labels = True,
        pos=pos,
        edge_color=edge_color,
        node_color=list(map(lambda x: node_colors[x], [nx.get_node_attributes(G,'frontier').get(n, False) for n in G.nodes()])),
    )
    if nx.get_edge_attributes(G, 'attn_weight') != {}:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'attn_weight'), font_size=6, alpha=0.8)
    plt.title(f"Selected {sum(np.array(edge_color) == 'green')} relevant edges")
    print(f"Selected {sum(np.array(edge_color) == 'green')} relevant edges over {len(G.edges())}")
    path = f'GOOD/kernel/pipelines/plots/{subfolder}/{config.load_split}_{config.util_model_dirname}/'
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            print(e)
            exit(e)
    plt.savefig(f'{path}/{name}.png')
    plt.close()
    return pos

def draw_topk(config, G, name, k, subfolder="", pos=None):
    if pos is None:
        pos = nx.kamada_kawai_layout(G)

    w = sorted(list(nx.get_edge_attributes(G, 'attn_weight').values()), reverse=True)
    edge_color = []
    for e in G.edges():
        if G.edges[e]["attn_weight"] >= w[k]:
            edge_color.append("green")
        else:
            edge_color.append("blue")

    nx.draw(
        G,
        with_labels = True,
        pos=pos,
        edge_color=edge_color
    )
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'attn_weight'), font_size=6, alpha=0.8)
    path = f'GOOD/kernel/pipelines/plots/{subfolder}/{config.load_split}_{config.util_model_dirname}/'
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            exit(e)
    plt.savefig(f'{path}/{name}.png')
    plt.close()
    return pos

def draw_gt(config, G, name, gt, edge_index, subfolder="", pos=None):
    if pos is None:
        pos = nx.kamada_kawai_layout(G)

    edge_color = {}
    for i in range(len(gt)):
        (u,v) = edge_index.T[i]
        if gt[i]:            
            edge_color[(u.item(), v.item())] = "green"
        else:
            edge_color[(u.item(), v.item())] = "blue"
    nx.draw(
        G,
        with_labels = True,
        pos=pos,
        edge_color=[edge_color[(u,v)] for u,v in G.edges()]
    )
    path = f'GOOD/kernel/pipelines/plots/{subfolder}/{config.load_split}_{config.util_model_dirname}/'
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            exit(e)
    plt.savefig(f'{path}/{name}.png')
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

def random_attach_no_target_frontier(S, T):
    # random attach frontier nodes in S and T
    # avoid selecting target nodes that are in the frontier

    S_frontier = list(filter(lambda x: nx.get_node_attributes(S,'frontier').get(x, False), S.nodes()))

    ret = nx.union(S, T, rename=("", "T"))
    for n in S_frontier:
        # pick random node v in G_t_spu
        # add edge (u,v) and (v,u)
        idx = randint(0, len(T.nodes()) - 1)
        v = "T" + str(list(T)[idx])
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
    # assert (tp + fp + tn + fn) == len(edge_gt)
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