import torch
from GOOD.data.good_datasets.good_cmnist import GOODCMNIST
from tqdm import tqdm
from torch_geometric.utils import to_networkx, from_networkx, to_undirected, sort_edge_index
from datetime import datetime
from mpi4py import MPI
from joblib import Parallel, delayed

# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()





dataset_root, domain,shift, generate = "/mnt/cimec-storage6/users/steve.azzolin/sedignn/leci_private_fork/storage/datasets", "color", "covariate", False

train_dataset = GOODCMNIST(root=dataset_root, domain=domain, shift=shift, subset='train', generate=generate, debias=True)
print(train_dataset)
print(train_dataset[0])

train_dataset = train_dataset[:10000]

print("Converting to nx")
startTime = datetime.now()
samples = []
for i , G in tqdm(enumerate(train_dataset), total=len(train_dataset)):
    samples.append(to_networkx(G))

# samples = Parallel(n_jobs=2)(delayed(to_networkx)(G) for G in train_dataset)
# print(samples[0])

print("Converting to PyG")
data_list = []
for i , G in tqdm(enumerate(samples), total=len(train_dataset)):
    data_list.append(from_networkx(G))

print("Completed in ", datetime.now() - startTime)