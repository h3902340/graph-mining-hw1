import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv


class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # two-layer GAT
        self.gat_layers.append(
            GATConv(
                in_size,
                hid_size,
                heads[0],
                feat_drop=0.3,
                activation=F.relu,
            )
        )
        self.gat_layers.append(
            GATConv(
                hid_size * heads[0],
                out_size,
                heads[1],
                activation=None,
            )
        )

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == len(self.gat_layers) - 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        return h
