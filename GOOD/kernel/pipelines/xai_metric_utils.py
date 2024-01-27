import networkx as nx
import torch
from random import randint, shuffle
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


def remove_from_graph(G, what_to_remove, edge_index_to_remove=None):
    if edge_index_to_remove is None:
        G = G.copy()
        edge_remove = []
        for (u,v), val in nx.get_edge_attributes(G, 'origin').items():
            if val == what_to_remove:
                edge_remove.append((u,v))
        G.remove_edges_from(edge_remove)
        G.remove_edges_from([(v,u) for v,u in G.edges() if not G.has_edge(u,v)])
        G.remove_nodes_from(list(nx.isolates(G)))
    else:
        G = G.copy()
        G.remove_edges_from([(u.item(), v.item()) for u,v in edge_index_to_remove.T])
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
        d = {(u.item(), v.item()): round(inv_edge_w[i].item(),2) for i, (u,v) in enumerate(inv_edge_index.T)}
        assert np.all([d[u,v] == d[v,u] for u,v in d.keys()])
        nx.set_edge_attributes(
            G,
            name="attn_weight",
            values=d
        )
    nx.set_edge_attributes(
        G,
        name="origin",
        values={(u.item(), v.item()): "spu" for u,v in spu_edge_index.T}
    )
    if not spu_edge_w is None:
        d = {(u.item(), v.item()): round(spu_edge_w[i].item(),2) for i, (u,v) in enumerate(spu_edge_index.T)}
        assert np.all([d[u,v] == d[v,u] for u,v in d.keys()])
        nx.set_edge_attributes(
            G,
            name="attn_weight",
            values=d
        )

def mark_frontier(G, G_filt):
    # mark frontier nodes as nodes attached to both inv and spu parts
    # to mark nodes check which nodes have a change in the degree between original and filtered graph
    # frontier = []
    # for n in G_filt.nodes():
    #     if G.degree[n] != G_filt.degree[n]:                    
    #         frontier.append(n)            
    
    frontier = list(filter(lambda n: G.degree[n] != G_filt.degree[n], G_filt.nodes()))

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
    plt.title(f"Selected {sum([e == 'green' for e in edge_color])} relevant edges")
    print(f"Selected {sum([e == 'green' for e in edge_color])} relevant edges over {len(G.edges())}")
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

        # assert str(n) in ret.nodes() and v in ret.nodes()

        ret.add_edge(str(n), v) #, origin="added"
        ret.add_edge(v, str(n)) #, origin="added"
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
        # assert str(n) in ret.nodes() and v in ret.nodes()

        ret.add_edge(str(n), v) #, origin="added"
        ret.add_edge(v, str(n)) #, origin="added"
    return ret

def expl_acc(expl, data, expl_weight=None):
    edge_gt = {(u.item(),v.item()): data.edge_gt[i] for i, (u,v) in enumerate(data.edge_index.T)} 
    edge_expl = set([(u.item(),v.item()) for u,v in expl.T])
    
    tp = int(sum([edge_gt[(u.item(),v.item())] for u,v in expl.T]))
    fp = int(sum([not edge_gt[(u.item(),v.item())] for u,v in expl.T]))
    tn = int(sum([not (u.item(),v.item()) in edge_expl and not edge_gt[(u.item(),v.item())] for u,v in data.edge_index.T]))
    fn = int(sum([not (u.item(),v.item()) in edge_expl and edge_gt[(u.item(),v.item())] for u,v in data.edge_index.T]))
    
    # acc = (tp + tn) / (tp + fp + tn + fn)
    f1 = 2*tp / (2*tp + fp + fn)
    # assert (tp + fp + tn + fn) == len(edge_gt)

    wiou, den = 0, 0
    for i, (u,v) in enumerate((data.edge_index.T)):
        u, v = u.item(), v.item()
        if edge_gt[(u,v)]:
            if (u,v) in edge_expl:
                wiou += expl_weight[i].item()
                den += expl_weight[i].item()
            else:
                den += expl_weight[i].item()
        elif (u,v) in edge_expl:
            den += expl_weight[i].item()
    wiou = wiou / den
    return round(wiou, 3), round(f1, 3)

def sample_edges(G_ori, where_to_sample, alpha, edge_index_to_remove=None):
    # keep each spu/inv edge with probability alpha
    G = G_ori.copy()
    if edge_index_to_remove is None:
        edges = set()
        for (u,v), val in nx.get_edge_attributes(G, 'origin').items():
            if val == where_to_sample:
                edges.add((u,v))
                # if where_to_sample == "spu" and np.random.binomial(1, alpha, 1)[0] == 0:
                #     edge_remove.append((u,v))
        edges = list(edges)
    else:
        edges = [(u.item(), v.item()) for u, v in edge_index_to_remove.T]    
    
    shuffle(edges)
    edge_remove = edges[:int(len(G.edges()) * (1-alpha))] #remove the 1-alpha% of the undirected edges
    G.remove_edges_from(edge_remove)
    G.remove_edges_from([(v,u) for v,u in G.edges() if not G.has_edge(u,v)])
    G.remove_nodes_from(list(nx.isolates(G)))
    return G

def feature_intervention(G, feature_bank, feat_int_alpha):
    """Randomly swap feature of spurious graph with features sampled from a fixed bank"""
    assert feature_bank .shape[0] > 0


    G = G.copy()
    probs = np.random.binomial(1, feat_int_alpha, len(G))
    for i, n in enumerate(G):
        if probs[i] == 1:
            new_feature = feature_bank[randint(0, feature_bank.shape[0]-1)].tolist()
            nx.set_node_attributes(G, {n: new_feature}, name="ori_x")
    return G
