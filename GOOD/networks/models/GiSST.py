import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import is_undirected, to_undirected, degree, coalesce
from torch_geometric import __version__ as pyg_v

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GINs import GINFeatExtractor
import copy
from GOOD.utils.splitting import split_graph, relabel


@register.model_register
class GiSSTGIN(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        
        # input_size,
        # output_size,
        # hidden_conv_sizes, 
        # hidden_dropout_probs, 
        # activation=F.relu,
        # classify_graph=False,
        # lin_dropout_prob=None,
        # **kwargs

        super(GiSSTGIN, self).__init__(config)

        config = copy.deepcopy(config)
        fe_kwargs = {'mitigation_readout': config.mitigation_readout}

        if config.mitigation_sampling == "raw":
            config.mitigation_backbone = None
            config.model.model_layer = 1

        self.gnn_clf = GINFeatExtractor(config)
        self.classifier = Classifier(config)

        self.prob_mask = ProbMask(config.dataset.dim_node)
        self.extractor = AttentionProb(config.dataset.dim_node)

        self.config = config
        self.edge_mask = None
        print("Using mitigation_expl_scores:", config.mitigation_expl_scores)
    
    def forward(self, *args, **kwargs):
        """
        Forward pass.

        Args:
            x (torch.float): Node feature tensor with shape [num_nodes, num_node_feat].
            edge_index (torch.long): Edges in COO format with shape [2, num_edges].
            edge_weight (torch.float): Weight for each edge with shape [num_edges].
            return_probs (boolean): Whether to return x_prob and edge_prob.
            batch (None or torch.long): Node assignment for a batch of graphs with shape 
                [num_nodes] for graph classification. None for node classification.

        Return:
            x (torch.float): Final output of the network with shape 
                [num_nodes, output_size].
            x_prob (torch.float): Node feature probability with shape [input_size].
            edge_prob (torch.float): Edge probability with shape [num_edges].
        """
        data = kwargs.get('data')

        # node feature explanation
        x_prob = self.prob_mask()
        x_prob.requires_grad_()
        x_prob.retain_grad()
        x = data.x * x_prob

        # edge topological explanation
        edge_prob = self.extractor(x, data.edge_index)
        edge_prob.requires_grad_()
        edge_prob.retain_grad()

        if is_undirected(data.edge_index):
            if self.config.average_edge_attn == "default":
                assert False, f"self.config.average_edge_attn == 'default' not in use"
            else:
                data.ori_edge_index = data.edge_index.detach().clone() #for backup and debug
                data.edge_index, edge_att = to_undirected(data.edge_index, edge_prob.squeeze(-1), reduce="mean")

                if not data.edge_attr is None:
                    edge_index_sorted, edge_attr_sorted = coalesce(data.ori_edge_index, data.edge_attr, is_sorted=False)
                    data.edge_attr = edge_attr_sorted    
        else:
            edge_att = edge_prob
        
        if kwargs.get('weight', None):
            if kwargs.get('is_ratio'):
                (causal_edge_index, causal_edge_attr, causal_edge_weight), _ = split_graph(data, edge_att, kwargs.get('weight'))
                causal_x, causal_edge_index, causal_batch, _ = relabel(x, causal_edge_index, data.batch)
                data.x = causal_x
                data.batch = causal_batch
                data.edge_index = causal_edge_index
                if not data.edge_attr is None:
                    data.edge_attr = causal_edge_attr
                edge_att = causal_edge_weight                
            else:
                data.edge_index = (data.edge_index.T[edge_att >= kwargs.get('weight')]).T
                if not data.edge_attr is None:
                    data.edge_attr = data.edge_attr[edge_att >= kwargs.get('weight')]
                edge_att = edge_att[edge_att >= kwargs.get('weight')]

        if self.config.mitigation_expl_scores == "topK" or self.config.mitigation_expl_scores == "topk":
            (causal_edge_index, causal_edge_attr, edge_att), \
                _ = split_graph(data, edge_att, self.config.mitigation_expl_scores_topk)
           
            causal_x, causal_edge_index, causal_batch, _ = relabel(x, causal_edge_index, data.batch)

            data_topk = Data(x=causal_x, edge_index=causal_edge_index, edge_attr=causal_edge_attr, batch=causal_batch)
            kwargs['data'] = data_topk
            kwargs["batch_size"] =  data.batch[-1].item() + 1
        
        
        set_masks(edge_att, self)
        logits = self.classifier(self.gnn_clf(*args, **kwargs))
        clear_masks(self)
        self.edge_mask = edge_att
        return logits, x_prob, edge_prob
        
class ProbMask(torch.nn.Module):
    """
    Probability mask generator.

    Args:
        shape (tuple of int): Shape of the probability mask.
        clamp_min (float): Clamping minimum for probability, for numerical stability.
        clamp_max (float): Clamping maximum for probability, for numerical stability. 
    """
    def __init__(
        self, 
        shape, 
        clamp_min=0.00001,
        clamp_max=0.99999
    ):
        super(ProbMask, self).__init__()
        self.shape = shape
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.mask_weight = torch.nn.Parameter(torch.randn(shape))

    def forward(self):
        return torch.clamp(
            torch.sigmoid(self.mask_weight), 
            self.clamp_min, 
            self.clamp_max
        )

class AttentionProb(torch.nn.Module):
    """
    Edge attention mechanism for generating sigmoid probability using concatentation of 
    source and target node features.

    Args:
        input_size (int): Number of input node features.
        clamp_min (float): Clamping minimum for the output probability, for numerical
            stability.
        clamp_max (float): Clamping maximum for the output probability, for numerical 
            stability.
    """
    def __init__(
        self, 
        input_size, 
        clamp_min=0.00001,
        clamp_max=0.99999
    ):
        super(AttentionProb, self).__init__()
        self.input_size = input_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.att_weight = torch.nn.Parameter(
            torch.randn(input_size * 2)
        )
    
    def forward(
        self,
        x,
        edge_index
    ):
        """
        Forward pass.

        Args:
            x (tensor): Node feature tensor with shape [num_nodes, input_size].
            edge_index (torch.long): edges in COO format with shape [2, num_edges].

        Return:
            att (tensor): Edge attention probability with shape [num_edges].
        """
        att = torch.matmul(
            torch.cat(
                (
                    x[edge_index[0, :], :], # source node features
                    x[edge_index[1, :], :]  # target node features
                ), 
                dim=1
            ), 
            self.att_weight
        )
        att = torch.sigmoid(att)
        att = torch.clamp(att, self.clamp_min, self.clamp_max)
        return att
    
def set_masks(mask: Tensor, model: nn.Module):
    r"""
    Modified from https://github.com/wuyxin/dir-gnn.
    """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            if pyg_v == "2.4.0":
                module._fixed_explain = True
            else:
                module.__explain__ = True
                module._explain = True
            module._apply_sigmoid = False    
            module.__edge_mask__ = mask
            module._edge_mask = mask


def clear_masks(model: nn.Module):
    r"""
    Modified from https://github.com/wuyxin/dir-gnn.
    """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            if pyg_v == "2.4.0":
                module._fixed_explain = False
            else:
                module.__explain__ = False
                module._explain = False
            module.__edge_mask__ = None
            module._edge_mask = None
