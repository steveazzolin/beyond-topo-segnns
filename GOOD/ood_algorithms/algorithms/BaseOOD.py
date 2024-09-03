"""
Base class for OOD algorithms
"""
from abc import ABC
from torch import Tensor
from torch_geometric.data import Batch
from torch_scatter import scatter_sum
from torch_scatter.composite import scatter_softmax

from GOOD.utils.config_reader import Union, CommonArgs, Munch
from typing import Tuple
from GOOD.utils.initial import reset_random_seed
from GOOD.utils.train import at_stage
import torch



class BaseOODAlg(ABC):
    r"""
    Base class for OOD algorithms

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(BaseOODAlg, self).__init__()
        self.optimizer: torch.optim.Adam = None
        self.scheduler: torch.optim.lr_scheduler._LRScheduler = None
        self.model: torch.nn.Module = None
        self.config = config

        self.mean_loss = None
        self.spec_loss = None
        self.stage = 0

    def stage_control(self, config):
        r"""
        Set valuables before each epoch. Largely used for controlling multi-stage training and epoch related parameter
        settings.

        Args:
            config: munchified dictionary of args.

        """
        if self.stage == 0 and at_stage(1, config):
            reset_random_seed(config)
            self.stage = 1

    def input_preprocess(self,
                         data: Batch,
                         targets: Tensor,
                         mask: Tensor,
                         node_norm: Tensor,
                         training: bool,
                         config: Union[CommonArgs, Munch],
                         **kwargs
                         ) -> Tuple[Batch, Tensor, Tensor, Tensor]:
        r"""
        Set input data format and preparations

        Args:
            data (Batch): input data
            targets (Tensor): input labels
            mask (Tensor): NAN masks for data formats
            node_norm (Tensor): node weights for normalization (for node prediction only)
            training (bool): whether the task is training
            config (Union[CommonArgs, Munch]): munchified dictionary of args

        Returns:
            - data (Batch) - Processed input data.
            - targets (Tensor) - Processed input labels.
            - mask (Tensor) - Processed NAN masks for data formats.
            - node_norm (Tensor) - Processed node weights for normalization.

        """
        return data, targets, mask, node_norm

    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
        r"""
        Process the raw output of model

        Args:
            model_output (Tensor): model raw output

        Returns (Tensor):
            model raw predictions

        """
        return model_output

    def loss_calculate(self, raw_pred: Tensor, targets: Tensor, mask: Tensor, node_norm: Tensor,
                       config: Union[CommonArgs, Munch], batch: Tensor = None) -> Tensor:
        r"""
        Calculate loss

        Args:
            raw_pred (Tensor): model predictions
            targets (Tensor): input labels
            mask (Tensor): NAN masks for data formats
            node_norm (Tensor): node weights for normalization (for node prediction only)
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.metric.loss_func()`, :obj:`config.model.model_level`)

        .. code-block:: python

            config = munchify({model: {model_level: str('graph')},
                                   metric: {loss_func: Accuracy}
                                   })


        Returns (Tensor):
            cross entropy loss

        """
        loss = config.metric.loss_func(raw_pred, targets, reduction='none') * mask
        loss = loss * node_norm * mask.sum() if config.model.model_level == 'node' else loss
        self.clf_loss = loss.detach().mean().item()
        
        # Penalize uncertain values of beta
        if self.config.global_side_channel in ("simple", "simple_filternode", "dt"):
            assert False
            p = self.model.beta.sigmoid()
            self.side_channel_loss = config.train.channel_int * (
                    - (p * torch.log(p + 1e-10) 
                    + (1 - p) * torch.log(1 - p + 1e-10))
                )
            loss += self.side_channel_loss

            # Penalize filter node feature attn
            if self.config.global_side_channel in ("simple_filternode", ):
                attn = self.global_filter_attn
                eps = 1e-6
                
                # Same loss as for edge weights, but is too strong as push everything to zero
                # attn_entropy_loss = (att * torch.log(att / r + eps) +
                #                         (1 - att) * torch.log((1 - att) / (1 - r + eps) + eps)).mean()

                # Entropy regularization
                # r = 0.01
                # attn_entropy_loss = - (att * torch.log(att + eps) 
                #                     + (1 - att) * torch.log(1 - att + eps))
                # loss += config.ood.ood_param * attn_entropy_loss.mean()

                # Entropy regularization over batch
                attn_norm_per_batch = scatter_softmax(attn.squeeze(1), batch)
                logattn = torch.log(attn_norm_per_batch + eps)
                self.entropy_filternode_loss = config.ood.ood_param * scatter_sum(-attn_norm_per_batch * logattn, batch).mean()
                loss += self.entropy_filternode_loss
        return loss
    

    def entropy_loss(logits, return_raw=False):
        logp = torch.log(logits + 0.0000000001)
        entropy = torch.sum(-logits * logp, dim=1)
        if not return_raw:
            entropy = torch.mean(entropy)
        return entropy

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch],
                         **kwargs) -> Tensor:
        r"""
        Process loss

        Args:
            loss (Tensor): base loss between model predictions and input labels
            data (Batch): input data
            mask (Tensor): NAN masks for data formats
            config (Union[CommonArgs, Munch]): munchified dictionary of args

        Returns (Tensor):
            processed loss

        """
        self.mean_loss = loss.sum() / mask.sum()
        return self.mean_loss

    def set_up(self, model: torch.nn.Module, config: Union[CommonArgs, Munch]):
        r"""
        Training setup of optimizer and scheduler

        Args:
            model (torch.nn.Module): model for setup
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.train.lr`, :obj:`config.metric`, :obj:`config.train.mile_stones`)

        Returns:
            None

        """
        self.model: torch.nn.Module = model
        if self.config.global_side_channel:
            print("Using gropus")
            self.optimizer = torch.optim.Adam(
                [
                    {'params': model.global_side_channel.parameters(), "lr": config.train.lr_filternode, 'weight_decay': config.train.channel_weight_decay},
                    {'params': model.combinator.parameters(), "lr": config.train.lr_filternode},
                    {'params': [p for name, p in model.named_parameters() if ('global_side_channel' not in name) and ('combinator' not in name)]}
                ],
                lr=config.train.lr,
                weight_decay=config.train.weight_decay
            )
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.train.lr,
                                            weight_decay=config.train.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=config.train.mile_stones,
                                                              gamma=0.1)

    def backward(self, loss):
        r"""
        Gradient backward process and parameter update.

        Args:
            loss: target loss
        """
        loss.backward()
        self.optimizer.step()
