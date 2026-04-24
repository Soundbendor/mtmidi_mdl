from collections import OrderedDict
from torch import nn
import os
import torch

class MLPProbe(nn.Module):
    def __init__(self, in_dim =4096, hidden_dims = [], out_dim = 10):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.initial_dropout = True
        self.num_hidden = len(hidden_dims)
        self.num_layers = self.num_hidden + 1

        cur_layers = []

        prev_dim = in_dim
        hidden_idx = 0
        relu_idx = 0
        for hidden_dim in hidden_dims:
            cur_layers.append( (f'linear_{hidden_idx}', nn.Linear(prev_dim, hidden_dim)) )
            cur_layers.append( (f'relu_{relu_idx}', nn.ReLU()) )
            prev_dim = hidden_dim
            hidden_idx += 1
            relu_idx += 1

        cur_layers.append( (f'linear_{hidden_idx}', nn.Linear(prev_dim, out_dim)) )
        self.layers = nn.Sequential(OrderedDict(cur_layers))

    def forward(self, x):
        out = self.layers(x)
        return out
