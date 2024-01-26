import torch
from torch_geometric.data import Data
from GOOD.utils.splitting import split_graph, relabel, sparse_sort, sparse_topk
from torch_geometric.utils import degree
from torch_geometric.utils import cumsum, scatter, softmax


def topK(
    x,
    ratio,
    batch,
    min_score,
    tol,
):
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter(x, batch, reduce='max')[batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero().view(-1)
        return perm

    if ratio >= 1.:
        return torch.arange(x.shape[0], device=x.device), torch.tensor([], dtype=torch.long)

    if ratio is not None:
        num_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce='sum')

        if ratio >= 1:
            k = num_nodes.new_full((num_nodes.size(0), ), int(ratio))
        else:
            k = (float(ratio) * num_nodes.to(x.dtype)).ceil().to(torch.long)

        x, x_perm = torch.sort(x.view(-1), descending=True)
        batch = batch[x_perm]
        batch, batch_perm = torch.sort(batch, descending=False, stable=True)

        arange = torch.arange(x.size(0), dtype=torch.long, device=x.device)
        ptr = cumsum(num_nodes)
        batched_arange = arange - ptr[batch]
        mask = batched_arange < k[batch]
        print(mask)

        return x_perm[batch_perm[mask]], x_perm[batch_perm[~mask]].sort()[0]

    raise ValueError("At least one of the 'ratio' and 'min_score' parameters "
                     "must be specified")




data = Data(
    x=torch.arange(10),
    edge_index=torch.tensor([
        [0, 1, 1, 2,   3, 4, 4, 5,],
        [1, 0, 2, 1,   4, 3, 5, 4,],
    ]),
    batch=torch.tensor([0,0,0,1,1,1])
)
edge_att = torch.tensor([1,1,0,1,  1,1,0,0])


sa1 = torch.tensor([[ 0,  0,  1,  1,  1,  1,  2,  2,  2,  3,  4,  5,  6,  7,  7,  7,  8,  8,
                        9,  9, 10, 10, 11, 11],
                    [ 1,  2,  0,  3,  4,  7,  0,  5,  6,  1,  1,  2,  2,  1,  8, 11,  7,  9,
                    8   , 10,  9, 11,  7, 10]])
sa2 = torch.tensor([[12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 16, 17, 18, 19, 19, 19, 20, 20,
                        21, 21, 22, 22, 23, 23],
                    [13, 14, 12, 15, 16, 19, 12, 17, 18, 13, 13, 14, 14, 13, 20, 23, 19, 21,
                        20, 22, 21, 23, 19, 22]])
sa = torch.cat([sa1, sa2], dim=1)

re1 = torch.tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9774, 1.0000, 0.9976, 0.9976,
                    1.0000, 1.0000, 0.9976, 0.9976, 0.9774, 1.0000, 1.0000, 1.0000, 1.0000,
                    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])
re2 = torch.tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9774, 1.0000, 0.9976, 0.9976,
                    1.0000, 1.0000, 0.9976, 0.9976, 0.9774, 1.0000, 1.0000, 1.0000, 1.0000,
                    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])
re = torch.cat([re1, re2], dim=0)

print(f"shape r1:", re1.shape)
print(f"shape r2:", re2.shape)

if re[0] < re[1]:
    assert False

index = torch.cat([torch.zeros_like(re1), torch.ones_like(re2)], dim=0).to(torch.int64)
src, perm = sparse_sort(re, index, descending=True)
print(sa, re)
print()
print(perm[:len(re1)])
print(perm[len(re1):])


num_nodes = degree(index, dtype=torch.long)
print(num_nodes)
k = (0.3 * num_nodes.to(float)).ceil().to(torch.long)
print(k)
start_indices = torch.cat([torch.zeros((1, ), device=src.device, dtype=torch.long), num_nodes.cumsum(0)])
mask = [torch.arange(k[i], dtype=torch.long, device=src.device) + start_indices[i] for i in range(len(num_nodes))]
mask = torch.cat(mask, dim=0)
mask = torch.zeros_like(index, device=index.device).index_fill(0, mask, 1).bool()
topk = perm[mask]

print(sa[:, topk])


# ! EASY DEBUG !
print("\n\n EASY DEBUG  ")
scores = torch.tensor([1., 0., 0.5, 1., 1.,  1., 0., 0.5,1., 1.,  1., 0., 0.5,1., 1.])
batch = torch.tensor([0,0,0,0,0,  1,1,1,1,1,  2,2,2,2,2])

x, perm, _, _, _ = sparse_topk(scores, batch, 0.2, descending=True)
print(x)
print(perm)




exit()




# ! FEY version !
print("\n\n FEY VERSION  ")
x_perm, excl = topK(re, 0.3, index, min_score=None, tol=1e-7)
print(x_perm)
print(excl)

print()
final = sa[:, x_perm]
print(final)

final = sa[:, excl]
print(final)












exit()
# ! Second version !
print("\n\n SECOND VERSION  ")

# Two sorts: the first one on the value,
# the second (stable) on the indices:
x, x_perm = torch.sort(re, dim=0, descending=True, stable=True)
print(x)
print(x_perm)

x_perm = x_perm[mask]

print("Masked x_perm")
print(x_perm)

sa = sa.take_along_dim(x_perm, dim=1)
index, index_perm = torch.sort(index, dim=dim, stable=True)
x = x.take_along_dim(index_perm, dim=dim)


exit()











(causal_edge_index, causal_edge_attr, causal_edge_weight), _ = split_graph(data, edge_att, 0.3)

print(data.batch[torch.unique(causal_edge_index)].unique(return_counts=True))

causal_x, causal_edge_index, causal_batch, _ = relabel(data.x, causal_edge_index, data.batch)

print("after")
print(data.batch[torch.unique(causal_edge_index)].unique(return_counts=True))

print(causal_batch.unique(return_counts=True))


