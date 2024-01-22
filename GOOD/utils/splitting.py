import torch
from torch_geometric.utils import degree

def split_graph(data, edge_score, ratio):
    has_edge_attr = hasattr(data, 'edge_attr') and getattr(data, 'edge_attr') is not None

    new_idx_reserve, new_idx_drop, _, _, _ = sparse_topk(edge_score, data.batch[data.edge_index[0]], ratio, descending=True)
    new_causal_edge_index = data.edge_index[:, new_idx_reserve]
    new_spu_edge_index = data.edge_index[:, new_idx_drop]

    new_causal_edge_weight = edge_score[new_idx_reserve]
    new_spu_edge_weight = - edge_score[new_idx_drop]

    if has_edge_attr:
        new_causal_edge_attr = data.edge_attr[new_idx_reserve]
        new_spu_edge_attr = data.edge_attr[new_idx_drop]
    else:
        new_causal_edge_attr = None
        new_spu_edge_attr = None

    return (new_causal_edge_index, new_causal_edge_attr, new_causal_edge_weight), \
        (new_spu_edge_index, new_spu_edge_attr, new_spu_edge_weight)
    
def sparse_sort(src: torch.Tensor, index: torch.Tensor, dim=0, descending=False, eps=1e-12):
    r'''
    Adopt from <https://github.com/rusty1s/pytorch_scatter/issues/48>_.
    '''
    f_src = src.float()
    f_min, f_max = f_src.min(dim)[0], f_src.max(dim)[0]
    norm = (f_src - f_min) / (f_max - f_min + eps) + index.float() * (-1) ** int(descending)
    perm = norm.argsort(dim=dim, descending=descending)

    return src[perm], perm


def sparse_topk(src: torch.Tensor, index: torch.Tensor, ratio: float, dim=0, descending=False, eps=1e-12):
    rank, perm = sparse_sort(src, index, dim, descending, eps)
    num_nodes = degree(index, dtype=torch.long)
    k = (ratio * num_nodes.to(float)).ceil().to(torch.long)
    start_indices = torch.cat([torch.zeros((1, ), device=src.device, dtype=torch.long), num_nodes.cumsum(0)])
    mask = [torch.arange(k[i], dtype=torch.long, device=src.device) + start_indices[i] for i in range(len(num_nodes))]
    mask = torch.cat(mask, dim=0)
    mask = torch.zeros_like(index, device=index.device).index_fill(0, mask, 1).bool()
    topk_perm = perm[mask] #topk edges selected
    exc_perm = perm[~mask]
    return topk_perm, exc_perm, rank, perm, mask

def relabel(x, edge_index, batch, pos=None):
    num_nodes = x.size(0)
    sub_nodes = torch.unique(edge_index)
    x = x[sub_nodes]
    batch = batch[sub_nodes]
    row, col = edge_index
    # remapping the nodes in the explanatory subgraph to new ids.
    node_idx = row.new_full((num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
    edge_index = node_idx[edge_index]
    if pos is not None:
        pos = pos[sub_nodes]
    return x, edge_index, batch, pos