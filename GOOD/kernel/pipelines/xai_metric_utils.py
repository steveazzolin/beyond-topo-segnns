import networkx as nx
import torch
from random import randint, shuffle
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import numpy as np
import os

from torch_geometric.utils import remove_isolated_nodes, dropout_edge
from torch_scatter import scatter_sum, scatter_add

from GOOD.utils.splitting import split_graph

edge_colors = {
    # "inv": "green",
    # "spu": "blue",
    "inv": "black",
    "spu": "green",
    "added": "red"
}
node_colors = {
    True: "red",
    False: "#1f78b4"
}


def remove_from_graph(G, edge_index_to_remove=None, what_to_remove=None):
    if edge_index_to_remove is None and what_to_remove:
        G = G.copy()
        edge_remove = []
        for (u,v), val in nx.get_edge_attributes(G, 'origin').items():
            if val == what_to_remove:
                edge_remove.append((u,v))
        G.remove_edges_from(edge_remove)
        G.remove_edges_from([(v,u) for v,u in G.edges() if not G.has_edge(u,v)])
        G.remove_nodes_from(list(nx.isolates(G)))
    elif not edge_index_to_remove is None:
        G = G.copy()
        G.remove_edges_from([(u.item(), v.item()) for u,v in edge_index_to_remove.T])
        G.remove_edges_from([(v,u) for v,u in G.edges() if not G.has_edge(u,v)])
        G.remove_nodes_from(list(nx.isolates(G)))
    else:
        raise ValueError(f"what_to_remove can not be None with edge_index None")
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

def draw(config, G, name, subfolder="", pos=None, save=True, figsize=(6.4, 4.8), nodesize=350, with_labels=True, title=None, ax=None):
    plt.figure(figsize=figsize)

    if pos is None:
        pos = nx.kamada_kawai_layout(G)

    # edge_color = list(map(lambda x: edge_colors[x], nx.get_edge_attributes(G,'origin').values()))
    node_gt = list(nx.get_node_attributes(G, "node_gt").values())
    edge_color = list(nx.get_edge_attributes(G, "attn_weight").values())
    edge_color = ["red" if e > 0.90 else "black" for e in edge_color]
    # nx.draw_networkx_edges(
    #     G,
    #     pos=pos,
    #     edge_color="black"
    # )
    nx.draw(
        G,
        with_labels=with_labels,
        pos=pos,
        ax=ax,
        node_size=nodesize,
        node_color = ['lightgreen' if node_gt[i] else 'orange' for i in range(len(node_gt))],
        # node_color=list(map(lambda x: node_colors[x], [nx.get_node_attributes(G,'frontier').get(n, False) for n in G.nodes()])),
        # edgelist=[e for i, e in enumerate(G.edges()) if edge_color[i] > 0.5],
        edge_color=edge_color,
        # edge_cmap=plt.cm.Reds,
    )

    # Annotate with edge scores
    if nx.get_edge_attributes(G, 'attn_weight') != {}:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'attn_weight'), font_size=6, alpha=0.8)

    # options = {
    #     "node_color":"#cccccc",
    #     "edge_color": colors,
    #     "width": 2,
    #     "edge_cmap": plt.cm.Reds,
    #     "with_labels": False,
    #     "node_size":25}
    # nx.draw_networkx(g,pos=pos,ax=ax,**options)
    
    title = title if title is not None else f"Selected {sum([e == 'green' for e in edge_color])} relevant edges"
    plt.title(title)
    print(f"Selected {sum([e == 'green' for e in edge_color])} relevant edges over {len(G.edges())}")

    if save:
        path = f'GOOD/kernel/pipelines/plots/{subfolder}/{config.load_split}_{config.util_model_dirname}/'
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except Exception as e:
                print(e)
                exit(e)
        plt.savefig(f'{path}/{name}.png')
    else:
        plt.show()

    if ax is None:
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

def draw_colored(config, G, name, subfolder="", pos=None, save=True, figsize=(6.4, 4.8), nodesize=350, with_labels=True, title=None, ax=None, thrs=0.9):
    plt.figure(figsize=figsize)

    if pos is None:
        pos = nx.kamada_kawai_layout(G)

    node_gt = list(nx.get_node_attributes(G, "node_gt").values())
    node_attr = list(nx.get_node_attributes(G, "x").values())
    
    node_colors = []
    for i in range(len(node_gt)):
        if node_gt[i]:
            node_colors.append("orange") # "lightgreen"
        elif node_attr[i] == [1.0, 0., 0.]:
            node_colors.append("red")
        else:
            node_colors.append("orange")
    
    edge_color = list(nx.get_edge_attributes(G, "attn_weight").values())
    edge_color = ["red" if e >= thrs else "black" for e in edge_color]

    nx.draw(
        G,
        with_labels=with_labels,
        pos=pos,
        ax=ax,
        node_size=nodesize,
        node_color=node_colors,
        edge_color=edge_color,
    )

    # Annotate with edge scores
    if nx.get_edge_attributes(G, 'attn_weight') != {}:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'attn_weight'), font_size=6, alpha=0.8)
    
    plt.title(title)

    if save:
        path = f'GOOD/kernel/pipelines/plots/{subfolder}/{config.load_split}_{config.util_model_dirname}_{config.random_seed}/'
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except Exception as e:
                print(e)
                exit(e)
        plt.savefig(f'{path}/{name}.png')
    else:
        plt.show()

    if ax is None:
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
    
    edge_attrs = list(nx.get_edge_attributes(S, "edge_attr").values())
    edge_gts = list(nx.get_edge_attributes(S, "edge_gt").values())
    S_frontier = list(filter(lambda x: nx.get_node_attributes(S,'frontier').get(x, False), S.nodes()))

    nx.set_edge_attributes(T, 0, "edge_gt") # mark every edge of target graph as not GT edge

    ret = nx.union(S, T, rename=("", "T"))
    for n in S_frontier:
        # pick random node v in G_t_spu
        # add edge (u,v) and (v,u)
        idx = randint(0, len(T.nodes()) - 1)
        v = "T" + str(list(T)[idx])
        # assert str(n) in ret.nodes() and v in ret.nodes()

        if edge_attrs != []:
            attr = randint(0, len(edge_attrs) - 1)
            ret.add_edge(str(n), v, edge_attr=edge_attrs[attr])
            ret.add_edge(v, str(n), edge_attr=edge_attrs[attr])
        elif edge_gts != []:
            ret.add_edge(str(n), v, edge_gt=0)
            ret.add_edge(v, str(n), edge_gt=0)
        else:
            ret.add_edge(str(n), v)
            ret.add_edge(v, str(n))
    return ret

def expl_acc(expl, data, expl_weight=None):
    edge_gt = {(u.item(),v.item()): data.edge_gt[i] for i, (u,v) in enumerate(data.edge_index.T)} 
    edge_expl = set([(u.item(),v.item()) for u,v in expl.T])
    
    # tp = int(sum([edge_gt[(u.item(),v.item())] for u,v in expl.T]))
    # fp = int(sum([not edge_gt[(u.item(),v.item())] for u,v in expl.T]))
    # tn = int(sum([not (u.item(),v.item()) in edge_expl and not edge_gt[(u.item(),v.item())] for u,v in data.edge_index.T]))
    # fn = int(sum([not (u.item(),v.item()) in edge_expl and edge_gt[(u.item(),v.item())] for u,v in data.edge_index.T]))
    
    # acc = (tp + tn) / (tp + fp + tn + fn)
    # f1 = 2*tp / (2*tp + fp + fn)
    f1 = 0.0
    # assert (tp + fp + tn + fn) == len(edge_gt)

    wiou, den = 0, 1e-12
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

def expl_acc_fast(expl, data, expl_weight=None):
    """
        Works under the assumption that expl=edge_index of the entire graph.
        This is because the stability_detector analysis is done via WIOU, which
        is evaluated over the entire graph, without the need to split into different
        ratios.
    """
    f1 = 0.0

    intersection = torch.sum(expl_weight[data.edge_gt == 1])
    union        = torch.sum(expl_weight)
    wiou_fast = intersection / union
    return torch.round(wiou_fast, decimals=3).item(), f1

def expl_acc_super_fast(batch_data, batch_edge_score, reference_intersection):
    """
        Works WITH BATCH OF DATA AND under the assumption that expl=edge_index of the entire graph.
        This is because the stability_detector analysis is done via WIOU, which
        is evaluated over the entire graph, without the need to split into different
        ratios.

        reference_intersection: What to consider to be defined the intersection of WIoU.
                                It can be either the GT explanation, resulting in Plausibility WIoU,
                                or the previously predicted hard explanaiton, resulting in Stability WIoU.
    """
    intersection = scatter_sum(batch_edge_score * reference_intersection, batch_data.batch[batch_data.edge_index[0]])
    union        = scatter_sum(batch_edge_score, batch_data.batch[batch_data.edge_index[0]])
    wiou_super_fast = intersection / (union + 1e-10)
    return torch.round(wiou_super_fast, decimals=3)

def sample_edges(G_ori, alpha, deconfounded, edge_index_to_remove):
    # keep each spu/inv edge with probability alpha
    G = G_ori.copy()
    if not deconfounded:
        # bernoulli sampling
        edges = set()
        for (u,v), val in nx.get_edge_attributes(G, 'origin').items():
            pass
            # if val == where_to_sample:
            #     edges.add((u,v))                
                # if where_to_sample == "spu" and np.random.binomial(1, alpha, 1)[0] == 0:
                #     edge_remove.append((u,v))
        edges = list(edges)
    else:
        edges = [(u.item(), v.item()) for u, v in edge_index_to_remove.T]    
    
    shuffle(edges)
    # edge_remove = edges[:int(len(G.edges()) * (1-alpha))] #remove the 1-alpha% of the undirected edges
    edge_remove = edges[:1]
    G.remove_edges_from(edge_remove)
    G.remove_edges_from([(v,u) for v,u in G.edges() if not G.has_edge(u,v)])
    G.remove_nodes_from(list(nx.isolates(G)))
    return G

def sample_edges_tensorized(data, nec_number_samples, sampling_type, nec_alpha_1, avg_graph_size, edge_index_to_remove=None, force_undirected=True):
    if sampling_type == "bernoulli":
        assert not nec_alpha_1 is None
        # customization of dropout_edge from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/dropout.html#dropout_edge
        data = data.clone()

        row, col = data.edge_index
        mask_noncausal = torch.ones(row.size(0), dtype=bool)
        mask_noncausal[edge_index_to_remove] = False
        
        edge_mask = torch.rand(row.size(0), device=data.edge_index.device) >= nec_alpha_1
        edge_mask[mask_noncausal] = True
        if force_undirected:
            edge_mask[row > col] = False

        edge_index = data.edge_index[:, edge_mask]

        if force_undirected:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_id_kept = edge_mask.nonzero().repeat((2, 1)).squeeze()
        else:
            edge_id_kept = edge_mask

        if hasattr(data, "edge_attr") and not data.edge_attr is None:
            data.edge_attr = data.edge_attr[edge_id_kept]
        
        data.edge_index, data.edge_attr, mask = remove_isolated_nodes(edge_index, data.edge_attr, num_nodes=data.x.shape[0])
        data.x = data.x[mask]
        data.num_nodes = data.x.shape[0]
        return data
    elif sampling_type == "deconfounded":
        if nec_number_samples == "prop_G_dataset":
            k = max(1, int(nec_alpha_1 * avg_graph_size))
        elif nec_number_samples == "prop_R":
            k = max(1, int(nec_alpha_1 * edge_index_to_remove.sum()))
            # if k == max(1, int(nec_alpha_1 * avg_graph_size)):
            #     print(k)
        elif nec_number_samples == "alwaysK":
            k = nec_alpha_1
        else:
            raise ValueError(f"value for nec_number_samples ({nec_number_samples}) not supported")
        
        k = min(k, int(data.edge_index.shape[1]/2)-2)
        assert k > 0, k

        data = data.clone()
        row, col = data.edge_index
        undirected = data.edge_index[:, row <= col]

        candidate_mask = edge_index_to_remove[row <= col]
        candidate_idxs = torch.argwhere(candidate_mask)
        
        # Version of the main paper with permutation (requires for loop)
        perm = torch.randperm(candidate_idxs.shape[0])
        to_keep = perm[:-k]
        removed = perm[-k:]

        causal_idxs_keep = candidate_idxs[to_keep].view(-1)
        causal_idxs_remove = candidate_idxs[removed].view(-1)

        to_keep = torch.zeros(undirected.shape[1], dtype=torch.bool)
        to_keep[candidate_mask == 0] = 1
        to_keep[causal_idxs_keep] = 1

        data.edge_index = torch.cat((undirected[:, to_keep], undirected[:, to_keep].flip(0)), dim=1)
        
        if hasattr(data, "edge_attr") and not data.edge_attr is None:
            undirected_edge_attr = data.edge_attr[row <= col]
            data.edge_attr = torch.cat((undirected_edge_attr[to_keep, :], undirected_edge_attr[to_keep, :]), dim=0)
        if hasattr(data, "edge_gt"):            
            undirected_edge_gt = data.edge_gt[row <= col]
            data.edge_gt = torch.cat((undirected_edge_gt[to_keep], undirected_edge_gt[to_keep]), dim=0)

        data.edge_index, data.edge_attr, mask = remove_isolated_nodes(data.edge_index, data.edge_attr, num_nodes=data.x.shape[0])
        data.x = data.x[mask]
        data.num_nodes = data.x.shape[0]
        return data
    else:
        raise ValueError(f"sampling_type {sampling_type} not valid")
    

def sample_edges_tensorized_batched(
        data,
        nec_number_samples,
        sampling_type, 
        nec_alpha_1,
        avg_graph_size,
        budget,
        edge_index_to_remove=None,
):
    if sampling_type == "bernoulli":
        raise NotImplementedError("")
    elif sampling_type == "deconfounded":
        if nec_number_samples == "prop_G_dataset":
            k = max(1, int(nec_alpha_1 * avg_graph_size))
        elif nec_number_samples == "prop_R":
            k = max(1, int(nec_alpha_1 * edge_index_to_remove.sum()))
        elif nec_number_samples == "alwaysK":
            k = nec_alpha_1
        else:
            raise ValueError(f"value for nec_number_samples ({nec_number_samples}) not supported")
        
        row, col = data.edge_index
        undirected = data.edge_index[:, row <= col]

        candidate_mask = edge_index_to_remove[row <= col]
        candidate_idxs = torch.argwhere(candidate_mask)
        
        k = min(k, int(data.edge_index.shape[1]/2)-2, candidate_idxs.shape[0])
        if k == 0:
            return None # None | [data.clone() for _ in range(budget)]

        # New version without perm, to avoid for loop
        random_weight_per_index = torch.rand(budget, candidate_idxs.shape[0], device=data.edge_index.device)
        topk_weight_per_index = torch.topk(random_weight_per_index, k=k, largest=True, dim=-1)
        
        all_except_topk = torch.ones(budget, candidate_idxs.shape[0], dtype=torch.bool)
        all_except_topk.scatter_(1, topk_weight_per_index.indices, False)

        to_keep = torch.arange(candidate_idxs.shape[0]).repeat(budget, 1)
        to_keep = to_keep.flatten()[all_except_topk.flatten()].reshape(to_keep.shape[0], all_except_topk.sum(-1)[0])
        # removed = topk_weight_per_index.indices

        causal_idxs_keep = candidate_idxs.reshape(1, -1).repeat(budget, 1).gather(1, to_keep) # B x elem_to_keep: indexes of edges to keep as elements
        # causal_idxs_remove = candidate_idxs.reshape(1, -1).repeat(budget, 1).gather(1, to_keep)

        to_keep = torch.zeros(budget, undirected.shape[1], dtype=torch.bool)
        to_keep[candidate_mask.repeat(budget, 1) == 0] = 1
        to_keep.scatter_(1, causal_idxs_keep, 1)

        intervened_graphs = []
        for k in range(budget):
            intervened_data = data.clone()
            intervened_data.edge_index = torch.cat((undirected[:, to_keep[k]], undirected[:, to_keep[k]].flip(0)), dim=1)
        
            if not (getattr(data, "edge_attr", None) is None):
                undirected_edge_attr = intervened_data.edge_attr[row <= col]
                intervened_data.edge_attr = torch.cat((undirected_edge_attr[to_keep[k], :], undirected_edge_attr[to_keep[k], :]), dim=0)
            if not (getattr(data, "edge_gt", None) is None):
                undirected_edge_gt = intervened_data.edge_gt[row <= col]
                intervened_data.edge_gt = torch.cat((undirected_edge_gt[to_keep[k]], undirected_edge_gt[to_keep[k]]), dim=0)
            if not (getattr(data, "causal_mask", None) is None):
                undirected_causal_mask = intervened_data.causal_mask[row <= col]
                intervened_data.causal_mask = torch.cat((undirected_causal_mask[to_keep[k]], undirected_causal_mask[to_keep[k]]), dim=0)

            intervened_data.edge_index, intervened_data.edge_attr, mask = remove_isolated_nodes(
                intervened_data.edge_index,
                intervened_data.edge_attr,
                num_nodes=intervened_data.x.shape[0]
            )
            if (~mask).sum() > 0: # at least one node was removed
                assert intervened_data.edge_index.shape[1] ==  intervened_data.edge_gt.shape[0], f"shape mismatch after remove_isolated_nodes(): {intervened_data.edge_index.shape[1]} vs {intervened_data.edge_gt.shape[0]}"
            intervened_data.x = intervened_data.x[mask]
            intervened_data.num_nodes = intervened_data.x.shape[0]
            intervened_graphs.append(intervened_data)
        return intervened_graphs
    else:
        raise ValueError(f"sampling_type {sampling_type} not valid")

def explanation_stability_hard(data, explanations, ratio):
    """
        Minimal example of the pitfall of F1:
        >>> from sklearn.metrics import f1_score, matthews_corrcoef
        >>> import numpy as np
        >>> true = np.array([0,1,1,1])
        >>> pred = np.array([1,1,1,1])
        >>> f1_score(true, pred, pos_label=1)
            0.8571428571428571
        >>> f1_score(true, pred, pos_label=0)
            0.0
        >>> matthews_corrcoef(true, pred)
            0.0
    """
    # Extract new hard explanation
    (causal_edge_index, _, _, causal_batch), \
        _, mask = split_graph(
            data,
            explanations,
            ratio,
            return_batch=True
    )     

    # (Slow version)
    # f1_single = []
    # for j in range(causal_batch.max() + 1):
    #     original_causal_mask = data.causal_mask[data.batch[data.edge_index[0]] == j].cpu()
    #     intervened_causal_mask = mask[data.batch[data.edge_index[0]] == j].cpu()
    #     f1_single.append(f1_score(original_causal_mask, intervened_causal_mask))

    # Calculating precision, recall, and F1 score using PyTorch (basic 1D case)
    # TP = ((input == 1) & (target == 1)).sum().item()
    # FP = ((input == 1) & (target == 0)).sum().item()
    # FN = ((input == 0) & (target == 1)).sum().item()

    # (Fast version)
    eps = 1e-6
    input = mask
    target = data.causal_mask

    # TODO: for interventions adding elements, manually add novel edges to the counts
    TP = scatter_add((input & target).to(int), data.batch[data.edge_index[0]], dim=0)
    TN = scatter_add(((input == False) & (target == False)).to(int), data.batch[data.edge_index[0]], dim=0)
    FP = scatter_add(((input == True) & (target == False)).to(int), data.batch[data.edge_index[0]], dim=0)
    FN = scatter_add(((input == False) & (target == True)).to(int), data.batch[data.edge_index[0]], dim=0)

    # Compute F1
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1_batched = 2 * (precision * recall) / (precision + recall + eps)

    # Compute MCC
    numerator = (TP * TN) - (FP * FN)
    denominator = torch.sqrt(((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    mcc_batched = numerator / (denominator + eps)
    return mcc_batched, f1_batched


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
