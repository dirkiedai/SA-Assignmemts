import math
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import src.utils as utils

class MultiLayerPerception(nn.Module):
    def __init__(
        self,
        cfg,
        ):
        super().__init__()

        self.num_layers = cfg.num_layers
        self.input_dim = cfg.input_dim
        self.output_dim = cfg.output_dim
        self.hidden_size = cfg.hidden_size
        self.dropout = cfg.dropout

        self.batch_norm = cfg.batch_norm
        self.residual_connection = cfg.residual_connection

        self.activate_fn = utils.get_activation_fn(cfg.activation_fn)

        self.project_in_dim = Linear(self.input_dim, self.hidden_size)

        self.activate_before = self.activate_fn()

        
        
        self.hidden_layers = nn.ModuleList([])
        self.hidden_layers.extend(
            [   nn.Sequential(
                Linear(self.hidden_size, self.hidden_size),
                self.activate_fn(),
                nn.Dropout(self.dropout),
            )
                for _ in range(self.num_layers)
            ]
        )
        self.project_out_dim = Linear(self.hidden_size, self.output_dim)

    @staticmethod
    def build_model(cls, args):
        pass
    
    def forward(
        self,
        source: Tensor,
        features_only: bool = False,
        return_all_hiddens: bool = False,
    ):
        x, extra = self.extract_features(source)

        if not features_only:
            # out_projection layer
            if self.project_out_dim is not None:
                x = self.project_out_dim(x)

        return (x, extra) if return_all_hiddens else x

        
    def extract_features(
        self,
        x: Tensor
    ):
        bsz, input_dim = x.size()
        assert input_dim == self.input_dim

        # in_projection layer
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)
            x = self.activate_before(x)

        inner_states: List[Optional[Tensor]] = [x]
        # hidden layers
        for layer in self.hidden_layers:
            x = x + layer(x) if self.residual_connection else layer(x)
            inner_states.append(x)
        return x, {"inner_states": inner_states}
        

def Linear(in_features, out_features, bias=True):
    layer = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(layer.weight)
    return layer

