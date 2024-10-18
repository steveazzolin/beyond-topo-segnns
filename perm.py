import torch
from torch_geometric.data import Data
from torch_geometric.utils import shuffle_node

# Example graph data
num_nodes = 5
num_edges = 6

# Random node features (e.g., 3-dimensional node feature)
x = torch.randn((num_nodes, 3))

# Edge indices (2 rows: source and destination of each edge)
edge_index = torch.tensor([[0, 1, 2, 3, 4, 0],
                           [1, 2, 3, 4, 0, 4]], dtype=torch.long)

# Edge attributes (e.g., 2-dimensional edge attribute)
edge_attr = torch.randn((num_edges, 2))

# Creating a sample PyTorch Geometric Data object
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# Step 1: Generate a random permutation for the nodes
# perm = torch.randperm(data.num_nodes)
_, perm = shuffle_node(data.x, torch.zeros(x.shape[0], dtype=torch.long))

# Step 2: Permute the node features
data.x = data.x[perm]


dict_perm = {p.item(): j for j, p in enumerate(perm)}
data.edge_index2 = torch.tensor([ [dict_perm[x.item()], dict_perm[y.item()]] for x,y in data.edge_index.T ]).T


# Step 3: Update the edge indices based on the node permutation
# Since edge_index is of shape [2, num_edges], we update both source and target indices
data.edge_index = perm[data.edge_index]

# Edge attributes remain unchanged as they are tied to the edges, not nodes
# (Edge attributes stay aligned with the edges that were permuted)
print("Permutation: \n", perm)
# print("Permuted node features:\n", data.x)
print("Permuted edge indices:\n", data.edge_index)
# print("Edge attributes:\n", data.edge_attr)

print("Permuted edge indices2:\n", data.edge_index2)
