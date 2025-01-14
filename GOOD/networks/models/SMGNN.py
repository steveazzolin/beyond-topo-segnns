import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import InstanceNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import is_undirected, to_undirected, degree, coalesce
from torch_sparse import transpose
from torch_geometric import __version__ as pyg_v

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .Classifiers import Classifier, ConceptClassifier, EntropyLinear
from .GINs import GINFeatExtractor, SimpleGlobalChannel, DecisionTreeGlobalChannel
from .GINvirtualnode import vGINFeatExtractor
import copy
from GOOD.utils.splitting import split_graph, relabel

@register.model_register
class SMGNNGIN(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(SMGNNGIN, self).__init__(config)
        
        config = copy.deepcopy(config)
        fe_kwargs = {'mitigation_readout': config.mitigation_readout}

        self.gnn = GINFeatExtractor(config, **fe_kwargs)
        self.extractor = ExtractorMLP(config)

        if config.mitigation_sampling == "raw":
            config.mitigation_backbone = None
            config.model.model_layer = 1
            self.gnn_clf = GINFeatExtractor(config)
        else:
            self.gnn_clf = None

        self.classifier = Classifier(config)
        self.learn_edge_att = config.ood.extra_param[0]
        self.config = config

        self.edge_mask = None
        print("Using mitigation_expl_scores:", config.mitigation_expl_scores)

        if config.global_side_channel in ("simple", "simple_filternode", "simple_product", "simple_productscaled", "simple_godel", "simple_linear"):
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
        elif config.global_side_channel == "simple_mlp":
            self.global_side_channel = SimpleGlobalChannel(config)
            self.beta = torch.nn.Parameter(data=torch.tensor(0.0), requires_grad=True)
            self.combinator = nn.Sequential(*(
                [
                    nn.Linear(config.dataset.num_classes * 2, 64, bias=False),
                    torch.nn.LeakyReLU(),
                    nn.Linear(64, 64),
                    torch.nn.LeakyReLU(),
                    nn.Linear(64, config.dataset.num_classes)
                ]
            ))
        elif config.global_side_channel == "dt":
            self.global_side_channel = DecisionTreeGlobalChannel(config)
            self.beta = torch.nn.Parameter(data=torch.tensor(0.0), requires_grad=True)

        if self.config.dataset.dataset_name == "MNIST":
            self.global_norm = InstanceNorm(config.dataset.num_classes * 2) #nn.functional.normalize #nn.BatchNorm1d(config.dataset.num_classes)

            # hidden_dim = 64
            # self.test = nn.Sequential(*(
            #     [
            #         # nn.Linear(config.dataset.num_classes * 2, hidden_dim, bias=False),
            #         EntropyLinear(config.dataset.num_classes * 2, hidden_dim, config.dataset.num_classes, bias=False, method=2, temperature=1.8),
            #         torch.nn.LeakyReLU(),
            #         nn.Linear(hidden_dim, hidden_dim),
            #         torch.nn.LeakyReLU(),
            #         # nn.Linear(hidden_dim, 1)
            #         nn.Linear(hidden_dim, config.dataset.num_classes) # WARNING: experimenting for Motif
            #     ]
            # ))
        

    def forward(self, *args, **kwargs):
        r"""
        The SMGNN model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            Label predictions and other results for loss calculations.

        """
        data = kwargs.get('data')
        
        emb = self.gnn(*args, without_readout=True, **kwargs)        
        att_log_logits = self.extractor(emb, data.edge_index, data.batch)
        # att = self.sampling(att_log_logits, self.training, self.config.mitigation_expl_scores)
        att = self.sampling(att_log_logits, False, self.config.mitigation_expl_scores)

        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                if self.config.average_edge_attn == "default":
                    nodesize = data.x.shape[0]
                    edge_att = (att + transpose(data.edge_index, att, nodesize, nodesize, coalesced=False)[1]) / 2
                else:
                    data.ori_edge_index = data.edge_index.detach().clone() #for backup and debug
                    data.edge_index, edge_att = to_undirected(data.edge_index, att.squeeze(-1), reduce="mean")

                    if not data.edge_attr is None:
                        edge_index_sorted, edge_attr_sorted = coalesce(data.ori_edge_index, data.edge_attr, is_sorted=False)
                        data.edge_attr = edge_attr_sorted    
            else:
                edge_att = att
        else:
            edge_att = self.lift_node_att_to_edge_att(att, data.edge_index)

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

        if self.config.mitigation_expl_scores == "topK" or self.config.mitigation_expl_scores == "topk":
            (causal_edge_index, causal_edge_attr, edge_att), \
                _ = split_graph(data, edge_att, self.config.mitigation_expl_scores_topk)
           
            causal_x, causal_edge_index, causal_batch, _ = relabel(data.x, causal_edge_index, data.batch)

            data_topk = Data(x=causal_x, edge_index=causal_edge_index, edge_attr=causal_edge_attr, batch=causal_batch)
            kwargs['data'] = data_topk
            kwargs["batch_size"] =  data.batch[-1].item() + 1

        set_masks(edge_att, self)

        if self.gnn_clf:
            logits = self.classifier(self.gnn_clf(*args, **kwargs))
        else:
            logits = self.classifier(self.gnn(*args, **kwargs))
        
        clear_masks(self)
        self.edge_mask = edge_att

        if self.config.global_side_channel and not kwargs.get('exclude_global', False):
            logits_side_channel, filter_attn = self.global_side_channel(**kwargs)
            logits_gnn = logits
            
            if "simple_concept" in self.config.global_side_channel:
                # mask_channel = torch.zeros_like(logits_side_channel)

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

                    temp = get_temp(start_temp=1, end_temp=self.config.train.end_temp, max_num_epoch=kwargs.get('max_num_epoch'), curr_epoch=kwargs.get('curr_epoch')) #if self.config.dataset.dataset_name != "MNIST" else 1
                    channel_gnn = torch.sigmoid(logits_gnn / temp)
                    channel_global = torch.sigmoid(logits_side_channel / temp)
                else:
                    channel_gnn = logits_gnn
                    channel_global = logits_side_channel
                        
                input = torch.cat((channel_gnn, channel_global), dim=1)
                if self.config.dataset.dataset_name == "MNIST":
                    input = self.global_norm(input)
                
                logits = self.combinator(input)                
            elif self.config.global_side_channel == "simple_product":
                # logits = logits_gnn.sigmoid() * logits_side_channel.sigmoid()
                # logits = torch.log(logits / (1 - logits + 1e-6)) # Revert Sigmoid
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
                # logits = torch.min(torch.cat((logits_gnn.sigmoid(), logits_side_channel.sigmoid()), dim=1), dim=1, keepdim=True).values
                logits = torch.min(logits_gnn.sigmoid(), logits_side_channel.sigmoid())
                logits = torch.log(logits / (1 - logits + 1e-6)) # Revert Sigmoid to logit space
            elif self.config.global_side_channel == "simple_linear" or self.config.global_side_channel == "simple_mlp":
                logits = self.combinator(torch.cat((logits_gnn.sigmoid(), logits_side_channel.sigmoid()), dim=1))
            else:
                # logits = self.beta.sigmoid() * logits_gnn + (1-self.beta.sigmoid()) * logits_side_channel
                # logits = self.beta.sigmoid() * logits_gnn.sigmoid() +  (1 - self.beta.sigmoid().detach()) * logits_side_channel.sigmoid().detach() # Combine them in probability space, and revert to logit for compliance with other code
                # logits = torch.log(logits / (1 - logits + 1e-10)) # Revert Sigmoid
                # Min(A,B)
                # logits = torch.min(torch.cat((logits_gnn, logits_side_channel), dim=1), dim=1, keepdim=True).values
                # Linear commbination
                # logits = self.combinator(torch.cat((logits_gnn.sigmoid(), logits_side_channel.sigmoid()), dim=1))
                exit("Not implemented")
            
            if torch.any(torch.isinf(logits)):
                print("Inf detected")
                # print(torch.exp(-logits_gnn)[:5].flatten(), torch.exp(-logits_side_channel)[:5].flatten(), torch.exp(-logits_gnn-logits_side_channel)[:5].flatten())
                idx = torch.isinf(logits)
                print(logits_gnn[idx].flatten(), logits_side_channel[idx].flatten())
                print(torch.exp(-logits_gnn)[idx].flatten(), torch.exp(-logits_side_channel)[idx].flatten(), torch.exp(-logits_gnn-logits_side_channel)[idx].flatten())
                exit("AIA")
            if torch.any(torch.isnan(logits)):
                print("NaN detected")
                print(logits_gnn[:5])
                print(edge_att[:5])
                exit("AIA")

            return logits, att_log_logits, att_log_logits.sigmoid(), filter_attn, (logits_gnn, logits_side_channel) # WARNING: I replaced edge_attn with att_log_logits.sigmoid()
        else:
            return logits, att_log_logits, att_log_logits.sigmoid() # WARNING: I replaced attn with att_log_logits

    def sampling(self, att_log_logits, training, mitigation_expl_scores):
        if mitigation_expl_scores == "anneal":
            temp = (self.config.train.epoch * 0.1 + (200 - self.config.train.epoch) * 5) / 200

        att = self.concrete_sample(att_log_logits, temp=1, training=training)
        if mitigation_expl_scores == "hard":
            att_hard = (att > 0.5).float()
            att = att_hard - att.detach() + att
        return att

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern
    
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
        set_masks(edge_att, self)
        if self.gnn_clf:
            logits = self.classifier(self.gnn_clf(*args, **kwargs))
        else:
            logits = self.classifier(self.gnn(*args, **kwargs))
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
    
    def get_subgraph(self, ratio=None, *args, **kwargs):
        data = kwargs.get('data') or None
        data.ori_x = data.x

        emb = self.gnn(*args, without_readout=True, **kwargs)
        att_log_logits = self.extractor(emb, data.edge_index, data.batch)
        att = self.sampling(att_log_logits, False, self.config.mitigation_expl_scores)

        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                if self.config.average_edge_attn == "default":
                    nodesize = data.x.shape[0]
                    edge_att = (att + transpose(data.edge_index, att, nodesize, nodesize, coalesced=False)[1]) / 2
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
        else:
            edge_att = self.lift_node_att_to_edge_att(att, data.edge_index)

        if kwargs.get('return_attn', False):
            self.attn_distrib = self.gnn.encoder.get_attn_distrib()
            self.gnn.encoder.reset_attn_distrib()

        edge_att = edge_att.view(-1)
        if ratio is None:
            return edge_att        
        assert False
        


@register.model_register
class SMGNNvGIN(SMGNNGIN):
    r"""
    The GIN virtual node version of SMGNN.
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(SMGNNvGIN, self).__init__(config)
        fe_kwargs = {'mitigation_readout': config.mitigation_readout}
        self.gnn = vGINFeatExtractor(config, **fe_kwargs)

        if config.mitigation_sampling == "raw":
            config.mitigation_backbone = None
            config.model.model_layer = 1
            self.gnn_clf = vGINFeatExtractor(config)
        else:
            self.gnn_clf = None


class ExtractorMLP(nn.Module):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__()
        hidden_size = config.model.dim_hidden
        self.learn_edge_att = config.ood.extra_param[0]  # learn_edge_att
        dropout_p = config.model.dropout_rate

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p)
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p)

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits


class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch):
        for module in self._modules.values():
            if isinstance(module, (InstanceNorm)):
                inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs


class MLP(BatchSequential):
    def __init__(self, channels, dropout, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                # m.append(InstanceNorm(channels[i])) # WARNING: Original GSAT was using this
                # m.append(nn.BatchNorm1d(channels[i]))
                m.append(nn.ReLU()) # WARNING: Original GSAT and first working GL-GSAT for best global channel was using ReLU
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)

#  goodtg --config_path final_configs/GOODMotif/basis/covariate/SMGNN.yaml --seeds "99" --task train --average_edge_attn mean --global_pool sum --gpu_idx 0 --global_side_channel "" --extra_param True 10 0.1 --ood_param 0.001 --lr_filternode 0.001 --lr 0.1 --train_bs 256

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
