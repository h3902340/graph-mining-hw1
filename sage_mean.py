import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv


class SAGE_MEAN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer SAGE
        self.layers.append(SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = self.dropout(features)
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h
