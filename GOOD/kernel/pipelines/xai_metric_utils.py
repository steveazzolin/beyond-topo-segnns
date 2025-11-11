import networkx as nx
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

node_colors = {
    True: "red",
    False: "#1f78b4"
}

def mark_edges(G, inv_edge_index, spu_edge_index, inv_edge_w=None, spu_edge_w=None):
    nx.set_edge_attributes(
        G,
        name="origin",
        values={(u.item(), v.item()): "inv" for u,v in inv_edge_index.T}
    )
    if not inv_edge_w is None:
        d = {(u.item(), v.item()): round(inv_edge_w[i].item(),2) for i, (u,v) in enumerate(inv_edge_index.T)}
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

def draw_colored(config, G, name, thrs, subfolder="", pos=None, save=True, figsize=(6.4, 4.8), nodesize=150, with_labels=True, title=None, ax=None):
    plt.figure(figsize=figsize)

    if pos is None:
        pos = nx.kamada_kawai_layout(G)

    node_gt = list(nx.get_node_attributes(G, "node_gt").values())
    node_attr = list(nx.get_node_attributes(G, "x").values())
    
    node_colors = []
    for i in range(len(node_attr)):
        if len(node_gt) > 0 and node_gt[i]:
            node_colors.append("orange")
        elif node_attr[i] == [1.0, 0., 0.]:
            node_colors.append("red")
        elif node_attr[i] == [1.0, 0.]:
            node_colors.append("red")
        elif node_attr[i] == [0., 1.]:
            node_colors.append("blue")
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
    # if nx.get_edge_attributes(G, 'attn_weight') != {}:
    #     nx.draw_networkx_edge_labels(
    #         G,
    #         pos,
    #         edge_labels=nx.get_edge_attributes(G, 'attn_weight'),
    #         font_size=6,
    #         alpha=0.8
    #     )
    
    plt.suptitle(title)

    if save:
        path = f'GOOD/kernel/pipelines/plots/{subfolder}/{config.load_split}_{config.util_model_dirname}_{config.random_seed}/'
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except Exception as e:
                print(e)
                exit(e)
        plt.savefig(f'{path}/{name}.pdf')
    else:
        plt.show()

    if ax is None:
        plt.close()
    return pos



def expl_acc(expl, data, expl_weight=None):
    edge_gt = {(u.item(),v.item()): data.edge_gt[i] for i, (u,v) in enumerate(data.edge_index.T)} 
    edge_expl = set([(u.item(),v.item()) for u,v in expl.T])
    f1 = 0.0

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

            intervened_data.num_edge_removed = data.edge_index.shape[1] - intervened_data.edge_index.shape[1]
            intervened_graphs.append(intervened_data)
        return intervened_graphs
    else:
        raise ValueError(f"sampling_type {sampling_type} not valid")
