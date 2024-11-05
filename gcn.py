import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv


class GCN(nn.Module):
    """
    Baseline Model:
    - A simple two-layer GCN model, similar to https://github.com/tkipf/pygcn
    - Implement with DGL package
    """

    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(GraphConv(in_size, hid_size, activation=F.relu))
        self.layers.append(GraphConv(hid_size, out_size))
        self.input_drop = nn.Dropout(0.2)
        self.dropout = nn.Dropout(0.2)
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
                h = F.relu(h)
                h = self.dropout(h)
        return h
