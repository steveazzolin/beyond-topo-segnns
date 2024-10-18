r"""
Applies a linear transformation to complete classification from representations.
"""
import torch
import torch.nn as nn
from torch import Tensor

import math

from GOOD.utils.config_reader import Union, CommonArgs, Munch


class Classifier(torch.nn.Module):
    r"""
    Applies a linear transformation to complete classification from representations.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.dataset.num_classes`)
    """
    def __init__(self, config: Union[CommonArgs, Munch], output_dim:int = None):

        super(Classifier, self).__init__()
        # self.classifier = nn.Sequential(*(
        #         [nn.Linear(config.model.dim_hidden, 2 * config.model.dim_ffn), nn.BatchNorm1d(2 * config.model.dim_ffn)] +
        #         [nn.ReLU(), nn.Linear(2 * config.model.dim_ffn, config.dataset.num_classes)]
        # ))
        self.classifier = nn.Sequential(*(
            [nn.Linear(config.model.dim_hidden, config.dataset.num_classes if output_dim is None else output_dim)]
        ))

    def forward(self, feat: Tensor) -> Tensor:
        r"""
        Applies a linear transformation to feature representations.

        Args:
            feat (Tensor): feature representations

        Returns (Tensor):
            label predictions

        """
        return self.classifier(feat)


class EntropyLinear(nn.Module):
    """
        Applies a linear transformation to the incoming data: :math:`y = xA^T + b` scaled by attention coefficients
        induced by parameter weight
    """

    def __init__(self, in_features: int, out_features: int, n_classes: int, temperature: float = 0.6,
                 bias: bool = True, remove_attention: bool = False, method=None) -> None:
        super(EntropyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_classes = n_classes
        self.temperature = temperature
        self.alpha = None
        self.remove_attention = remove_attention
        self.method = method
        self.has_bias = bias

        n_classes = 1 # WARNING: experimenting for Motif
        self.weight = nn.Parameter(torch.Tensor(n_classes, out_features, in_features))
        if method is None:
            self.gamma = nn.Parameter(torch.randn((n_classes, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_classes, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        if len(input.shape) == 2:
            input = input.unsqueeze(0)
        
        # compute concept-awareness scores
        if self.method == 2:
            self.gamma = self.weight.norm(dim=1, p=1)
        self.alpha = torch.exp(self.gamma/self.temperature) / torch.sum(torch.exp(self.gamma/self.temperature), dim=1, keepdim=True)

        if self.method == 2:
            self.alpha_norm = self.alpha
        else:
            # Avoid numerical cancellations due to values close to zero
            self.alpha_norm = self.alpha / self.alpha.max(dim=1)[0].unsqueeze(1)
        
        if self.remove_attention:
            self.concept_mask = torch.ones_like(self.alpha_norm, dtype=torch.bool)
            x = input
        else:
            # weight the input concepts by awareness scores
            self.concept_mask = self.alpha_norm > 0.5
            x = input.multiply(self.alpha_norm.unsqueeze(1))

        # compute linear map
        x = x.matmul(self.weight.permute(0, 2, 1))
        if self.has_bias:
             x += self.bias
        return x.permute(1, 0, 2).squeeze(1)
    
    
class ConceptClassifier(torch.nn.Module):
    r"""
    """
    def __init__(self, config: Union[CommonArgs, Munch], method=None):

        super(ConceptClassifier, self).__init__()

        hidden_dim = config.dataset.num_classes * 2 * 5
        self.classifier = nn.Sequential(*(
            [
                EntropyLinear(config.dataset.num_classes * 2, hidden_dim, config.dataset.num_classes, bias=False, method=method),
                torch.nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                torch.nn.LeakyReLU(),
                # nn.Linear(hidden_dim, 1)
                nn.Linear(hidden_dim, config.dataset.num_classes) # WARNING: experimenting for Motif
            ]
        ))
        self.config = config

    def forward(self, feat: Tensor) -> Tensor:
        r"""
        Applies a linear transformation to feature representations.

        Args:
            feat (Tensor): feature representations

        Returns (Tensor):
            label predictions

        """
        out = self.classifier(feat)
        # if self.config.dataset.num_classes > 1:
        #     out = out.squeeze(2)
        return out