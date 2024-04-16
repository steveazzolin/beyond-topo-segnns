import torch
import torch_geometric
import networkx as nx
import matplotlib.pyplot as plt
from GOOD.data.good_datasets.good_sst2 import GOODSST2
from GOOD.data.good_datasets.good_motif2 import GOODMotif2

from torch_geometric.utils import from_networkx, to_networkx


# dataset = GOODSST2(
#     root="/mnt/cimec-storage6/users/steve.azzolin/sedignn/leci_private_fork/storage/datasets",
#     domain="length",
#     shift="covariate",
#     subset='train',
#     generate=False
# )
# print(dataset)
# print(dataset[0])



dataset = GOODMotif2(
    root="/mnt/cimec-storage6/users/steve.azzolin/sedignn/leci_private_fork/storage/datasets",
    domain="basis",
    shift="covariate",
    subset='test',
    generate=False
)
print(dataset)
print(dataset[0])
g = to_networkx(dataset[1])
nx.draw(g)
plt.savefig("graph.png")
