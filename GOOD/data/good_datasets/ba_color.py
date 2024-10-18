import os.path as osp
import random

import torch
from munch import Munch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx, shuffle_node, barabasi_albert_graph, erdos_renyi_graph

from sklearn.model_selection import train_test_split

from GOOD import register
from GOOD.utils.synthetic_data.BA3_loc import *

import networkx as nx
import random


@register.dataset_register
class BAColor(InMemoryDataset):
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

        if shift == "no_shift":
            self.num_nodes_min = 10
            self.num_nodes_max = 100
            self.graph_distribution = "BA"
        elif shift == "size":
            self.num_nodes_min = 150
            self.num_nodes_max = 250
            self.graph_distribution = "BA"
        elif shift == "ER":
            self.num_nodes_min = 10
            self.num_nodes_max = 100
            self.graph_distribution = "ER"
        elif shift == "color":
            self.num_nodes_min = 10
            self.num_nodes_max = 100
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

        super(BAColor, self).__init__(root, transform, pre_transform)

        # self.data, self.slices = self.process()
        print("loading: ", self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self):
        data_list = []

        for _ in range(self.num_graphs):
            # Step 1: Generate a random number of nodes
            if self.domain == "basis":
                num_blue_nodes = random.randint(self.num_nodes_min // 2, self.num_nodes_max // 2)
                num_red_nodes = random.randint(self.num_nodes_min // 2, self.num_nodes_max // 2)

                x = torch.cat(
                    (
                        torch.tensor([[0., 1.]]).repeat(num_blue_nodes, 1),
                        torch.tensor([[1., 0.]]).repeat(num_red_nodes, 1)
                    ),
                    dim=0
                )
                y = torch.tensor([[0.] if num_red_nodes >= num_blue_nodes else [1.]])
            elif self.domain == "color":
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

            # Step 2: Generate a graph
            if self.graph_distribution == "BA":
                edge_index = barabasi_albert_graph(num_nodes=num_blue_nodes + num_red_nodes, num_edges=2)
            elif self.graph_distribution == "ER":
                edge_index = erdos_renyi_graph(num_nodes=num_blue_nodes + num_red_nodes, edge_prob=0.2, directed=False)

            # Step 3: Assign random node features (color 0 or 1) with one-hot encoding
            perm = torch.randperm(x.shape[0])
            x = x[perm]

            # Step 6: Create a Data object
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        # Step 7: Collate data objects into a dataset
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
        return [f'data_{self.graph_distribution}_numgraphs{self.num_graphs}_min{self.num_nodes_min}_max{self.num_nodes_max}_shift{self.shift}.pt']
    
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

        if domain == "basis":
            dataset = BAColor(dataset_root, domain=domain)
            ood1_dataset = BAColor(dataset_root, domain=domain, shift="size")
            ood2_dataset = BAColor(dataset_root, domain=domain, shift="ER")
        elif domain == "color":
            dataset = BAColor(dataset_root, domain=domain)
            ood1_dataset = BAColor(dataset_root, domain=domain, shift="size")
            ood2_dataset = BAColor(dataset_root, domain=domain, shift="color")
        elif domain == "constant":
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
            meta_info.num_classes = train_dataset._data.y.shape[1]
        elif train_dataset.task == 'Regression':
            meta_info.num_classes = 1
        elif train_dataset.task == 'Multi-label classification':
            meta_info.num_classes = torch.unique(train_dataset._data.y).shape[0]

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
