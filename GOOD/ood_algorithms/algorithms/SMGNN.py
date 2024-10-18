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
class SMGNN(BaseOODAlg):
    r"""
    Implementation of the SMGNN algorithm

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(SMGNN, self).__init__(config)
        self.att = None
        self.edge_att = None
        self.decay_r = 0.1
        self.decay_interval = config.ood.extra_param[1]
        self.final_r = config.ood.extra_param[2]

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
            raw_out, self.att, self.edge_att, self.global_filter_attn, (self.logit_gnn, self.logit_global) = model_output
        else:
            raw_out, self.att, self.edge_att = model_output
        return raw_out

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch], epoch:int,
                         **kwargs) -> Tensor:
        r"""
        Process loss based on GSAT algorithm

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
            loss based on SMGNN algorithm

        """
        att = self.edge_att 

        # L1 sparsification
        self.l_norm_loss = self.config.train.l_norm_coeff * att.squeeze(1).abs().mean(-1) # L1
        # self.l_norm_loss = att.squeeze(1).pow(2).mean(-1) # L2        
        # Entropy regularization
        attn = att.squeeze(1)
        self.entr_loss = self.config.train.entr_coeff * torch.mean(-attn * torch.log(attn + 1e-6) - (1 - attn) * torch.log(1 - attn + 1e-6))
        info_loss = self.l_norm_loss + self.entr_loss

        if epoch < 10: # pre-train phase; 10 just for Motif
            self.spec_loss = torch.tensor(0., device=att.device)
        else:
            self.spec_loss = config.ood.ood_param * info_loss

        if self.att.mean() < -20:
            eps = 1e-6
            r = 0.1
            # correction_loss = (self.att * torch.log(self.att / r + eps) + (1 - self.att) * torch.log((1 - self.att) / (1 - r + eps) + eps)).mean()
            correction_loss = -self.att.mean() # Push them towards zero
            self.spec_loss += 0.1 * correction_loss
            print(f"*****************Collapse detected (correction_loss = {correction_loss:.3f}) {self.att.max()} {self.att.mean()}**********************")

        self.mean_loss = loss.mean()
        loss = self.mean_loss + self.spec_loss
        return loss
    
    def loss_global_side_channel(self, targets: Tensor, mask: Tensor, config: Union[CommonArgs, Munch]) -> Tensor:
        loss = config.metric.loss_func(self.logit_global, targets, reduction='none') * mask
        return loss.mean()
