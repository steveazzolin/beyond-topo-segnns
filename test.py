import torch
from torch_sparse import coalesce
from torch_geometric.utils import coalesce as coal

index = torch.tensor([[1, 2, 0, 1, 2],
                      [0, 1, 1, 2, 1]])
value = torch.Tensor([1,2,3,4, 10])

index2, value2 = coalesce(index, value, m=3, n=3)

print(index2)
print(value2)

index, value = coal(index, value, num_nodes=3, is_sorted=False)

print()
print(index)
print(value)