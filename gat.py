import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv


class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
        # two-layer GAT
        self.conv1 = GATConv(in_size, hid_size, heads[0])
        self.conv2 = GATConv(hid_size * heads[0], out_size, heads[1])
        

    def forward(self, g, h):
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.conv1(g, h)
        h = F.elu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.conv2(g, h)
        return h
