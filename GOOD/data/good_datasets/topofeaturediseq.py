import os.path as osp
import random

import torch
from munch import Munch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx, shuffle_node, barabasi_albert_graph, erdos_renyi_graph, to_networkx, contains_isolated_nodes

from sklearn.model_selection import train_test_split

from GOOD import register
from GOOD.utils.synthetic_data.BA3_loc import *
from GOOD.utils.synthetic_data.synthetic_structsim import dircycle

import networkx as nx
import random


@register.dataset_register
class TopoFeatureDiseq(InMemoryDataset):
    r"""
        A simple graph where:
        - class 1 iff motif AND at least N red nodes
        - class 0 otherwise
    """
    def __init__(self, root: str, domain: str = 'basis', shift: str = 'no_shift', subset: str = 'train', transform=None,
                 pre_transform=None, generate: bool = False, debias=False):
        
        self.name = self.__class__.__name__
        self.domain = domain
        self.shift = shift
        self.minority_class = None
        self.metric = 'Accuracy'
        self.task = 'Binary classification'
        self.url = ''
        
        self.num_graphs = 5000
        self.required_num_nodes_red = 5 # Num nodes required for class 1 # WARNING: changed from 2 to 10

        if shift == "no_shift":
            self.num_nodes_min = 8
            self.num_nodes_max = 80
            self.num_nodes_red_max = 10
            self.graph_distribution = "BA"
        elif shift == "size":
            self.num_nodes_min = 150
            self.num_nodes_max = 250
            self.num_nodes_red_max = 80
            self.graph_distribution = "BA"
        elif shift == "ER":
            self.num_nodes_min = 8
            self.num_nodes_max = 80
            self.num_nodes_red_max = 10
            self.graph_distribution = "ER"
        elif shift == "color":
            self.num_nodes_min = 8
            self.num_nodes_max = 80
            self.num_nodes_red_max = 10
            self.graph_distribution = "BA"
            self.colors = torch.tensor([
                [1., 1., 1.],
                [0., 0., 0.],
                [1., 1., 0.],
                [0., 1., 1.],
                [1., 0., 1.],
                [0., 0., 1.],
            ]) # some of these colors might not be adversarial since they do not change the relative number of RED and BLUE
        else:
            raise NotImplementedError(f"{shift} shift not implemented")

        super(TopoFeatureDiseq, self).__init__(root, transform, pre_transform)

        print("loading: ", self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self):
        data_list = []

        for _ in range(self.num_graphs):
            # Step 1: Generate a random number of nodes
            if self.domain == "basis":
                pattern = random.randint(0, 2)

                if pattern == 1:
                    if random.randint(0, 1) == 0:
                        pattern = np.random.choice(np.array([0, 2]))

                if pattern == 0:
                    # Include R >= B but do not include the motif
                    y = torch.tensor([[0]], dtype=torch.float)
                    
                    num_blue_nodes = random.randint(self.num_nodes_min // 2, self.num_nodes_max // 2)
                    num_red_nodes = random.randint(num_blue_nodes, self.num_nodes_max // 2 + 1)
                elif pattern == 1:
                    # Include the motif but R < B
                    y = torch.tensor([[0]], dtype=torch.float)

                    num_blue_nodes = random.randint(self.num_nodes_min // 2 + 1, self.num_nodes_max // 2)
                    num_red_nodes = random.randint(self.num_nodes_min // 2, num_blue_nodes - 1)
                else:
                    # Include the motif AND R >= B
                    y = torch.tensor([[1]], dtype=torch.float)

                    num_blue_nodes = random.randint(self.num_nodes_min // 2, self.num_nodes_max // 2)
                    num_red_nodes = random.randint(num_blue_nodes, self.num_nodes_max // 2 + 1)

                x = torch.cat(
                    (
                        torch.tensor([[1., 0., 0.]]).repeat(num_red_nodes, 1),
                        torch.tensor([[0., 1., 0.]]).repeat(num_blue_nodes, 1)
                    ),
                    dim=0
                )
            elif self.domain == "color":
                raise NotImplementedError("Not yet adapted")
                num_other_nodes = random.randint(0, 40) # min 2 max 25 other nodes
                num_blue_nodes = random.randint(self.num_nodes_min // 2, self.num_nodes_max // 2 - num_other_nodes)
                num_red_nodes = random.randint(self.num_nodes_min // 2, self.num_nodes_max // 2 - num_other_nodes)
                
                if self.shift == "color": #add more complex color combinations
                    color_combinations = [random.randint(0, self.colors.shape[0] - 1) for _ in range(num_other_nodes)]
                    x = torch.cat(
                        (
                            torch.tensor([[0., 1., 0.]]).repeat(num_blue_nodes, 1),
                            torch.tensor([[1., 0., 0.]]).repeat(num_red_nodes, 1),
                            self.colors[color_combinations]
                        ),
                        dim=0
                    )
                else:
                    x = torch.cat(
                        (
                            torch.tensor([[0., 1., 0.]]).repeat(num_blue_nodes, 1),
                            torch.tensor([[1., 0., 0.]]).repeat(num_red_nodes, 1),
                            torch.tensor([[0., 0., 1.]]).repeat(num_other_nodes, 1)
                        ),
                        dim=0
                    )
                y = torch.tensor([[0.] if num_red_nodes >= num_blue_nodes else [1.]])
            elif self.domain == "constant":
                raise NotImplementedError("Not yet adapted")
                num_blue_nodes = random.randint(self.num_nodes_min // 2, self.num_nodes_max // 2)
                num_red_nodes = random.randint(self.num_nodes_min // 2, self.num_nodes_max // 2)
                x = torch.cat(
                    (
                        torch.tensor([[0., 1.]]).repeat(num_blue_nodes, 1),
                        torch.tensor([[1., 0.]]).repeat(num_red_nodes, 1)
                    ),
                    dim=0
                )
                y = torch.tensor([[0.] if num_red_nodes >= (self.num_nodes_max - self.num_nodes_min) // 2 else [1.]])

            # Generate a base graph
            if self.graph_distribution == "BA":
                basis = nx.barabasi_albert_graph(n=num_red_nodes + num_blue_nodes, m=2)
            elif self.graph_distribution == "ER":
                basis = nx.erdos_renyi_graph(n=num_red_nodes + num_blue_nodes, p=0.2, directed=False)

            # Assign random node features (color 0 or 1) with one-hot encoding
            perm = torch.randperm(x.shape[0])
            x = x[perm]

            # Add the motif if needed
            if pattern == 1 or pattern == 2:
                motif, roles_motif = eval("dircycle")(start=num_red_nodes + num_blue_nodes, role_start=num_red_nodes + num_blue_nodes)
                plugin = np.random.choice(num_red_nodes + num_blue_nodes, 1, replace=False) # Attach point
                
                graph = basis.copy()
                graph.add_nodes_from(motif.nodes())
                graph.add_edges_from([(roles_motif[0], plugin[0])])
                graph.add_edges_from(motif.edges())
                additional_edge = 1 # Accounting for the link between basis and motif

                x = torch.cat(
                    (
                        x,
                        # torch.tensor([[1., 1., 1.]]).repeat(motif.number_of_nodes(), 1),
                        torch.tensor([[0., 0., 0.]]).repeat(motif.number_of_nodes(), 1), # Simplified version
                    ),
                    dim=0
                )
            else:
                motif = nx.Graph()
                graph = basis
                additional_edge = 0

            # Create a Data object
            data = Data(
                x=x,
                edge_index=from_networkx(graph).edge_index,
                y=y,
                pattern = torch.tensor([pattern], dtype=torch.long),
                node_gt=torch.tensor([0]*basis.number_of_nodes() + [1]*motif.number_of_nodes(), dtype=torch.bool),
                edge_gt=torch.tensor([0]*(basis.number_of_edges() + additional_edge)*2 + [1]*motif.number_of_edges()*2, dtype=torch.bool), # *2 because undirected
            )            
            data_list.append(data)

        # Collate data objects into a dataset
        data, slices = self.collate(data_list)
        print("Saving data in: ", self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])
        return data, slices

    def len(self):
        return self.num_graphs
    
    @property
    def raw_dir(self):
        return osp.join(self.root)
    
    @property
    def raw_file_names(self):
        # Since we're generating data, we don't have raw files
        return []

    def download(self):
        # No download needed since the data is generated on the fly
        pass

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, self.domain, 'processed')

    @property
    def processed_file_names(self):
        return [
            f'data_{self.graph_distribution}_numgraphs{self.num_graphs}' \
            f'_min{self.num_nodes_min}_max{self.num_nodes_max}' \
            f'_shift{self.shift}.pt'
        ]
    
    @staticmethod
    def load(dataset_root: str, domain: str= 'basis', shift: str = 'no_shift', generate: bool = False, debias: bool =False):
        r"""
        A staticmethod for dataset loading. This method instantiates dataset class, constructing train, id_val, id_test,
        ood_val (val), and ood_test (test) splits. Besides, it collects several dataset meta information for further
        utilization.

        Args:
            dataset_root (str): The dataset saving root.
            domain (str): The domain selection. Allowed: 'degree' and 'time'.
            shift (str): The distributional shift we pick. Allowed: 'no_shift', 'covariate', and 'concept'.
            generate (bool): The flag for regenerating dataset. True: regenerate. False: download.

        Returns:
            dataset or dataset splits.
            dataset meta info.
        """
        assert domain in ["basis", "color"] and shift == "no_shift", f"{domain} - {shift} not supported"
        meta_info = Munch()
        meta_info.dataset_type = 'syn'
        meta_info.model_level = 'graph'

        print("TopoFeatureDiseq")

        if domain == "basis":
            dataset = TopoFeatureDiseq(dataset_root, domain=domain)
            ood1_dataset = TopoFeatureDiseq(dataset_root, domain=domain, shift="size")
            ood2_dataset = TopoFeatureDiseq(dataset_root, domain=domain, shift="ER")
        elif domain == "color":
            assert False
            dataset = BAColor(dataset_root, domain=domain)
            ood1_dataset = BAColor(dataset_root, domain=domain, shift="size")
            ood2_dataset = BAColor(dataset_root, domain=domain, shift="color")
        elif domain == "constant":
            assert False
            dataset = BAColor(dataset_root, domain=domain)
            ood1_dataset = dataset #BAColor(dataset_root, domain=domain, shift="size")
            ood2_dataset = dataset #BAColor(dataset_root, domain=domain, shift="color")

        index_train, index_val_test = train_test_split(
            torch.arange(len(dataset)), 
            train_size=0.8,
            stratify=dataset.y,
            random_state=42
        )
        index_val, index_test = train_test_split(
            torch.arange(len(dataset[index_val_test])), 
            train_size=0.5,
            stratify=dataset[index_val_test].y,
            random_state=42
        )

        train_dataset = dataset[index_train]
        id_val_dataset = dataset[index_val]
        id_test_dataset = dataset[index_test]
        val_dataset = ood1_dataset
        test_dataset = ood2_dataset

        meta_info.dim_node = train_dataset.num_node_features
        meta_info.dim_edge = train_dataset.num_edge_features

        meta_info.edge_feat_dims = 0
        meta_info.num_envs = 1

        # Define networks' output shape.
        if train_dataset.task == 'Binary classification':
            meta_info.num_classes = train_dataset.data.y.shape[1]
        elif train_dataset.task == 'Regression':
            meta_info.num_classes = 1
        elif train_dataset.task == 'Multi-label classification':
            meta_info.num_classes = torch.unique(train_dataset.data.y).shape[0]

        train_dataset.minority_class = None
        id_val_dataset.minority_class = None
        id_test_dataset.minority_class = None
        val_dataset.minority_class = None
        test_dataset.minority_class = None
        train_dataset.metric = 'Accuracy'
        id_val_dataset.metric = 'Accuracy'
        id_test_dataset.metric = 'Accuracy'
        val_dataset.metric = 'Accuracy'
        test_dataset.metric = 'Accuracy'

        # --- clear buffer dataset._data_list ---        
        train_dataset._data_list = None
        if id_val_dataset:
            id_val_dataset._data_list = None
            id_test_dataset._data_list = None
            val_dataset._data_list = None
            test_dataset._data_list = None

        return {'train': train_dataset, 'id_val': id_val_dataset, 'id_test': id_test_dataset,
                'metric': 'Accuracy', 'task': dataset.task,
                'val': val_dataset, 'test': test_dataset}, meta_info


# goodtg --config_path final_configs/TopoFeature/basis/no_shift/GSAT.yaml --seeds "1" --task train --average_edge_attn mean --global_pool sum --gpu_idx 1 --global_side_channel simple_filternode --extra_param True 20 0.75 --wandb

# import matplotlib.pyplot as plt
            
# G = to_networkx(data, node_attrs=["x"])
# node_attr = list(nx.get_node_attributes(G, "x").values())
# node_colors = ["red" if e == [1.0, 0., 0.] else "blue" for e in node_attr]

# nx.draw(G, with_labels=True, node_color=node_colors)
# path = f'GOOD/kernel/pipelines/'
# plt.savefig(f'{path}/delete_me.png')
# plt.close()