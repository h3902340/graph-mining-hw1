import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv


class GCN_THREE(nn.Module):
    """
    Baseline Model:
    - A simple two-layer GCN model, similar to https://github.com/tkipf/pygcn
    - Implement with DGL package
    """

    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # 3-layer GCN
        self.conv1 = GraphConv(in_size, hid_size, activation=F.relu)
        self.conv2 = GraphConv(hid_size, hid_size)
        self.conv3 = GraphConv(hid_size, out_size)

    def forward(self, g, features):
        h = self.conv1(g, features)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        return h
