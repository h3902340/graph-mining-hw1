import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv


class SAGE(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # Two-layer GraphSAGE-gcn.
        self.layers.append(SAGEConv(in_size, hidden_size, "mean"))
        self.layers.append(SAGEConv(hidden_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)

    def forward(self, graph, x):
        hidden_x = x
        for layer_idx, layer in enumerate(self.layers):
            hidden_x = layer(graph, hidden_x)
            is_last_layer = layer_idx == len(self.layers) - 1
            if not is_last_layer:
                hidden_x = F.relu(hidden_x)
                hidden_x = self.dropout(hidden_x)
        return hidden_x
