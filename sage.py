import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv


class SAGE(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # Two-layer GraphSAGE-gcn.
        self.layers.append(SAGEConv(in_size, hidden_size, "gcn"))
        self.layers.append(SAGEConv(hidden_size, out_size, "gcn"))
        self.input_drop = nn.Dropout(0.25)
        self.dropout = nn.Dropout(0.75)
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(in_size, hidden_size))
        self.linears.append(nn.Linear(hidden_size, out_size))
        self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, graph, feat):
        h = feat
        h = self.input_drop(h)
        for i, layer in enumerate(self.layers):
            conv = layer(graph, h)
            linear = self.linears[i](h)
            h = conv + linear
            is_last_layer = i == len(self.layers) - 1
            if not is_last_layer:
                h = self.bn(h)
                h = F.relu(h)
                h = self.dropout(h)
        """hidden_x = x
        for layer_idx, layer in enumerate(self.layers):
            hidden_x = layer(graph, hidden_x)
            is_last_layer = layer_idx == len(self.layers) - 1
            if not is_last_layer:
                hidden_x = F.relu(hidden_x)
                hidden_x = self.dropout(hidden_x)"""
        return h
