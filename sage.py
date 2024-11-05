import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv, GraphConv


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # Two-layer GraphSAGE-gcn.
        self.layers.append(SAGEConv(in_size, hid_size, "gcn"))
        self.layers.append(SAGEConv(hid_size, out_size, "gcn"))
        self.input_drop = nn.Dropout(0.25)
        self.dropout = nn.Dropout(0.75)
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(in_size, hid_size))
        self.linears.append(nn.Linear(hid_size, out_size))
        self.bn = nn.BatchNorm1d(hid_size)

    def forward(self, g, features):
        h = features
        h = self.input_drop(h)
        for i, layer in enumerate(self.layers):
            conv = layer(g, h)
            linear = self.linears[i](h)
            h = conv + linear
            is_last_layer = i == len(self.layers) - 1
            if not is_last_layer:
                h = self.bn(h)
                h = F.leaky_relu(h)
                h = self.dropout(h)
        return h
