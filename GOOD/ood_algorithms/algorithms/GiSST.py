"""
Implementation of the GSAT algorithm from `"Interpretable and Generalizable Graph Learning via Stochastic Attention Mechanism" <https://arxiv.org/abs/2201.12987>`_ paper
"""
from typing import Tuple

import torch
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_sum
from torch_scatter.composite import scatter_softmax

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from GOOD.utils.initial import reset_random_seed
from GOOD.utils.train import at_stage
from .BaseOOD import BaseOODAlg


@register.ood_alg_register
class GiSST(BaseOODAlg):
    r"""
    Implementation of the GiSST algorithm

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GiSST, self).__init__(config)
        self.edge_att = None
        self.feat_att = None
        self.edge_l1 = config.ood.extra_param[1]
        self.edge_entr = config.ood.extra_param[2]
        self.feat_l1 = config.ood.extra_param[3]
        self.feat_entr = config.ood.extra_param[4]

    def stage_control(self, config: Union[CommonArgs, Munch]):
        r"""
        Set valuables before each epoch. Largely used for controlling multi-stage training and epoch related parameter
        settings.

        Args:
            config: munchified dictionary of args.

        """
        if self.stage == 0 and at_stage(1, config):
            reset_random_seed(config)
            self.stage = 1

    def output_postprocess(self, model_output: Tensor, return_edge_scores: bool = False, **kwargs) -> Tensor:
        r"""
        Process the raw output of model

        Args:
            model_output (Tensor): model raw output

        Returns (Tensor):
            model raw predictions.

        """
        if len(model_output) == 5:
            raw_out, self.feat_att, self.edge_att, self.global_filter_attn, (self.logit_gnn, self.logit_global) = model_output
        else:
            raw_out, self.feat_att, self.edge_att = model_output
        return raw_out

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch], epoch:int,
                         **kwargs) -> Tensor:
        r"""
        Process loss based on GiSST algorithm

        Args:
            loss (Tensor): base loss between model predictions and input labels
            data (Batch): input data
            mask (Tensor): NAN masks for data formats
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)
            epoch (int): number of current epoch

        .. code-block:: python

            config = munchify({device: torch.device('cuda'),
                                   dataset: {num_envs: int(10)},
                                   ood: {ood_param: float(0.1)}
                                   })


        Returns (Tensor):
            loss based on GiSST algorithm

        """ 
        edge_att = self.edge_att
        feat_att = self.feat_att

        # L1 sparsification
        self.edge_l1_loss = self.edge_l1 * edge_att.abs().mean(-1)
        self.feat_l1_loss = self.feat_l1 * feat_att.abs().mean(-1)
        
        # Entropy regularization
        self.edge_entr_loss = self.edge_entr * torch.mean(-edge_att * torch.log(edge_att + 1e-6) - (1 - edge_att) * torch.log(1 - edge_att + 1e-6))
        self.feat_entr_loss = self.feat_entr * torch.mean(-feat_att * torch.log(feat_att + 1e-6) - (1 - feat_att) * torch.log(1 - feat_att + 1e-6))

        self.mean_loss = loss.mean()
        loss = self.mean_loss + self.edge_entr_loss + self.edge_l1_loss + self.feat_l1_loss + self.feat_entr_loss
        return loss
    
