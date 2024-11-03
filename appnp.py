import argparse
import time

import dgl
import numpy as np
from dgl.data import (
    CiteseerGraphDataset,
    CoraGraphDataset,
    PubmedGraphDataset,
    register_data_args,
)
from dgl.nn.pytorch.conv import APPNPConv
from mxnet import gluon, nd
from mxnet.gluon import nn

class APPNP(nn.Block):
    def __init__(
        self,
        in_feats,
        hiddens,
        n_classes,
        activation,
        feat_drop,
        edge_drop,
        alpha,
        k,
    ):
        super(APPNP, self).__init__()
        with self.name_scope():
            self.layers = nn.Sequential()
            # input layer
            self.layers.add(nn.Dense(hiddens[0], in_units=in_feats))
            # hidden layers
            for i in range(1, len(hiddens)):
                self.layers.add(nn.Dense(hiddens[i], in_units=hiddens[i - 1]))
            # output layer
            self.layers.add(nn.Dense(n_classes, in_units=hiddens[-1]))
            self.activation = activation
            if feat_drop:
                self.feat_drop = nn.Dropout(feat_drop)
            else:
                self.feat_drop = lambda x: x
            self.propagate = APPNPConv(k, alpha, edge_drop)

    def forward(self, g, features):
        # prediction step
        h = features
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        h = self.layers[-1](self.feat_drop(h))
        # propagation step
        h = self.propagate(g, h)
        return h