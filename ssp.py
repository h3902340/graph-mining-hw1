import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch.conv import CFConv


class SSP(nn.Module):
    def __init__(self, node_in, edge_in, hid_size, out_size, heads):
        super().__init__()
        # two-layer GAT
        self.conv1 = CFConv(node_in, edge_in, hid_size, out_size)
        

    def forward(self, g, h):
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.conv1(g, h)
        h = F.elu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.conv2(g, h)
        return h
