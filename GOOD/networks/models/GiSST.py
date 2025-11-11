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
from .Classifiers import Classifier, ConceptClassifier
from .GINs import GINFeatExtractor, SimpleGlobalChannel, DecisionTreeGlobalChannel
import copy
from GOOD.utils.splitting import split_graph, relabel


@register.model_register
class GiSSTGIN(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
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

        if config.global_side_channel in ("simple", "simple_filternode", "simple_product", "simple_productscaled", "simple_godel"):
            self.global_side_channel = SimpleGlobalChannel(config)
            self.beta = torch.nn.Parameter(data=torch.tensor(0.0), requires_grad=True)
            self.combinator = nn.Linear(config.dataset.num_classes*2, config.dataset.num_classes, bias=True) # not in use
        elif config.global_side_channel == "simple_concept":
            self.global_side_channel = SimpleGlobalChannel(config)
            self.beta = torch.tensor(torch.nan)
            self.combinator = ConceptClassifier(config)
        elif config.global_side_channel == "simple_concept2":
            self.global_side_channel = SimpleGlobalChannel(config)
            self.beta = torch.tensor(torch.nan)
            self.combinator = ConceptClassifier(config, method=2)
        elif config.global_side_channel in ("simple_concept2discrete", "simple_concept2temperature"):
            self.global_side_channel = SimpleGlobalChannel(config)
            self.beta = torch.tensor(torch.nan)
            self.combinator = ConceptClassifier(config, method=2)
        elif config.global_side_channel == "dt":
            self.global_side_channel = DecisionTreeGlobalChannel(config)
            self.beta = torch.nn.Parameter(data=torch.tensor(0.0), requires_grad=True)

    
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

        if self.config.dataset.dataset_type == 'mol':
            # if node features are needed to compute AtomEmbedding, apply attention later only to extract edge scores
            scaled_x = data.x * x_prob
        else:
            data.x = data.x * x_prob
            scaled_x = data.x

        # edge topological explanation
        edge_log_prob = self.extractor(scaled_x, data.edge_index)
        
        edge_prob = torch.sigmoid(edge_log_prob)
        edge_prob = torch.clamp(edge_prob, self.extractor.clamp_min, self.extractor.clamp_max)
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
                causal_x, causal_edge_index, causal_batch, _ = relabel(data.x, causal_edge_index, data.batch)
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
        
        set_masks(edge_att, self)
        logits = self.classifier(self.gnn_clf(*args, **kwargs))
        clear_masks(self)
        self.edge_mask = edge_att

        if self.config.global_side_channel and not kwargs.get('exclude_global', False):
            logits_side_channel, filter_attn = self.global_side_channel(**kwargs)
            logits_gnn = logits
            
            if "simple_concept" in self.config.global_side_channel:
                if self.config.global_side_channel == "simple_concept2discrete":
                    # discrete reparametrization trick
                    if logits_gnn.shape[1] > 1:
                        index = logits_gnn.max(-1, keepdim=True)[1]
                        logits_gnn_hard = torch.zeros_like(logits_gnn).scatter_(-1, index, 1.0)    
                        index = logits_side_channel.max(-1, keepdim=True)[1]
                        logits_side_channel_hard = torch.zeros_like(logits_side_channel).scatter_(-1, index, 1.0)    
                    else:
                        logits_gnn_hard = (logits_gnn >= 0.).to(torch.float)
                        logits_side_channel_hard = (logits_side_channel >= 0.).to(torch.float)
                    
                    channel_gnn = logits_gnn_hard - logits_gnn.detach() + logits_gnn
                    channel_global = logits_side_channel_hard - logits_side_channel.detach() + logits_side_channel
                elif self.config.global_side_channel == "simple_concept2temperature":
                    def get_temp(start_temp, end_temp, max_num_epoch, curr_epoch):
                        if max_num_epoch is None:
                            return end_temp
                        if curr_epoch <= 20:
                            return start_temp
                        return start_temp - (start_temp - end_temp) / max_num_epoch * curr_epoch

                    temp = get_temp(start_temp=1, end_temp=self.config.train.end_temp, max_num_epoch=kwargs.get('max_num_epoch'), curr_epoch=kwargs.get('curr_epoch'))
                    channel_gnn = torch.sigmoid(logits_gnn / temp)
                    channel_global = torch.sigmoid(logits_side_channel / temp)
                else:
                    channel_gnn = logits_gnn
                    channel_global = logits_side_channel
                        
                logits = self.combinator(torch.cat((channel_gnn, channel_global), dim=1))
            elif self.config.global_side_channel == "simple_product":
                logits_gnn = torch.clip(logits_gnn, min=-50, max=50)
                logits_side_channel = torch.clip(logits_side_channel, min=-50, max=50)
                # logits_gnn = torch.full_like(logits_side_channel, 50) # masking one of the two channels setting to TRUE
                logits = -torch.log(torch.exp(-logits_gnn) + torch.exp(-logits_side_channel) + torch.exp(-logits_gnn-logits_side_channel) + 1e-6) # Invert product of sigmoids in log space
            elif self.config.global_side_channel == "simple_productscaled":
                logits_gnn = torch.clip(logits_gnn, min=-20, max=20)
                logits_side_channel = torch.clip(logits_side_channel, min=-20, max=20)
                # logits_gnn = torch.full_like(logits_side_channel, 20) # masking one of the two channels setting to TRUE
                logits = -torch.log(torch.exp(-logits_gnn/0.5) + torch.exp(-logits_side_channel/0.5) + torch.exp(-logits_gnn/0.5-logits_side_channel/0.5) + 1e-6) # Invert product of sigmoids in log space
            elif self.config.global_side_channel == "simple_godel":
                logits_gnn = torch.clip(logits_gnn, min=-50, max=50)
                logits_side_channel = torch.clip(logits_side_channel, min=-50, max=50)
                # logits_gnn = logits_side_channel # masking one channel
                logits = torch.min(logits_gnn.sigmoid(), logits_side_channel.sigmoid())
                logits = torch.log(logits / (1 - logits + 1e-6))
            else:
                raise NotImplementedError(f"{self.config.global_side_channel} not implemented for SMGNN")
            return logits, x_prob, edge_prob, filter_attn, (logits_gnn, logits_side_channel)
        else:
            return logits, x_prob, edge_prob
    
    @torch.no_grad()
    def probs(self, *args, **kwargs):
        # nodes x classes
        out = self(*args, **kwargs)
        
        if len(out) == 5:
            logits, att, edge_att, _, _ = out
        else:
            logits, att, edge_att = out

        if logits.shape[-1] > 1:
            return logits.softmax(dim=1)
        else:
            return logits.sigmoid()
        
    @torch.no_grad()
    def log_probs(self, eval_kl=False, *args, **kwargs):
        # nodes x classes
        out = self(*args, **kwargs)
        
        if len(out) == 5:
            logits, att, edge_att, _, _ = out
        else:
            logits, att, edge_att = out
            
        if logits.shape[-1] > 1:
            return logits.log_softmax(dim=1)
        else:
            if eval_kl: # make the single logit a proper distribution summing to 1 to compute KL
                logits = logits.sigmoid()
                new_logits = torch.zeros((logits.shape[0], logits.shape[1]+1), device=logits.device)
                new_logits[:, 1] = new_logits[:, 1] + logits.squeeze(1)
                new_logits[:, 0] = 1 - new_logits[:, 1]
                new_logits[new_logits == 0.] = 1e-10
                return new_logits.log()
            else:
                return logits.sigmoid().log()
            
    @torch.no_grad()
    def predict_from_subgraph(self, edge_att=False, log=None, eval_kl=None,  *args, **kwargs):
        # node feature explanation
        x_prob = self.prob_mask()
        x_prob.requires_grad_()
        x_prob.retain_grad()
        kwargs['data'].x = kwargs['data'].x * x_prob

        set_masks(edge_att, self)
        logits = self.classifier(self.gnn_clf(*args, **kwargs))
        clear_masks(self)

        if self.config.global_side_channel == "simple_concept2temperature":
            logits_side_channel, filter_attn = self.global_side_channel(**kwargs)
            logits_gnn = logits           
                
            def get_temp(start_temp, end_temp, max_num_epoch, curr_epoch):
                if max_num_epoch is None:
                    return end_temp
                if curr_epoch <= 20:
                    return start_temp
                return start_temp - (start_temp - end_temp) / max_num_epoch * curr_epoch

            temp = get_temp(start_temp=1, end_temp=self.config.train.end_temp, max_num_epoch=kwargs.get('max_num_epoch'), curr_epoch=kwargs.get('curr_epoch'))
            channel_gnn = torch.sigmoid(logits_gnn / temp)
            channel_global = torch.sigmoid(logits_side_channel / temp)
                
            lc_logits = self.combinator(torch.cat((channel_gnn, channel_global), dim=1))
        else:
            raise NotImplementedError("FIX ME")

        if log is None:
            if lc_logits.shape[-1] > 1:
                return lc_logits.argmax(-1)
            else:
                return lc_logits.sigmoid()
        else:
            assert not (eval_kl is None)
            if lc_logits.shape[-1] > 1:
                return lc_logits.log_softmax(dim=1)
            else:
                if eval_kl: # make the single logit a proper distribution summing to 1 to compute KL
                    lc_logits = lc_logits.sigmoid()
                    new_logits = torch.zeros((lc_logits.shape[0], lc_logits.shape[1]+1), device=lc_logits.device)
                    new_logits[:, 1] = new_logits[:, 1] + lc_logits.squeeze(1)
                    new_logits[:, 0] = 1 - new_logits[:, 1]
                    new_logits[new_logits == 0.] = 1e-10
                    return new_logits.log()
                else:
                    return lc_logits.sigmoid().log()

    @torch.no_grad()         
    def get_subgraph(self, ratio=None, *args, **kwargs):
        data = kwargs.get('data') or None
        data.ori_x = data.x

        # node feature explanation
        x_prob = self.prob_mask()
        data.x = data.x * x_prob

        # edge topological explanation
        att = self.extractor(data.x, data.edge_index)
        att_log_prob = self.extractor(data.x, data.edge_index)
        att = torch.sigmoid(att_log_prob)
        att = torch.clamp(att, self.extractor.clamp_min, self.extractor.clamp_max)
        
        if is_undirected(data.edge_index):
            if self.config.average_edge_attn == "default":
                raise NotImplementedError("average_edge_attn='default' not implemented")
            else:
                data.ori_edge_index = data.edge_index.detach().clone() #for backup and debug

                if not data.edge_attr is None:
                    edge_index_sorted, edge_attr_sorted = coalesce(data.ori_edge_index, data.edge_attr, is_sorted=False)
                    data.edge_attr = edge_attr_sorted   
                if hasattr(data, "edge_gt") and not data.edge_gt is None:
                    edge_index_sorted, edge_gt_sorted = coalesce(data.ori_edge_index, data.edge_gt, is_sorted=False)
                    data.edge_gt = edge_gt_sorted
                if hasattr(data, "causal_mask") and not data.causal_mask is None:
                    _, data.causal_mask = coalesce(data.edge_index, data.causal_mask, is_sorted=False)

                data.edge_index, edge_att = to_undirected(data.edge_index, att.squeeze(-1), reduce="mean")
        else:
            edge_att = att
        

        if kwargs.get('return_attn', False):
            self.attn_distrib = self.gnn.encoder.get_attn_distrib()
            self.gnn.encoder.reset_attn_distrib()

        edge_att = edge_att.view(-1)
        if ratio is None:
            return edge_att        
        assert False
        
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
