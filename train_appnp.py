import argparse
import time
import torch
import dgl

from data_loader import load_data
import mxnet as mx
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
        g,
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
        self.g = g

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

    def forward(self, features):
        # prediction step
        h = features
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        h = self.layers[-1](self.feat_drop(h))
        # propagation step
        h = self.propagate(self.g, h)
        return h


def evaluate(model, features, labels, mask):
    pred = model(features).argmax(axis=1)
    accuracy = ((pred == labels) * mask).sum() / len(mask)
    return accuracy


def main():
    # load and preprocess dataset
    (
        features,
        graph,
        n_classes,
        train_labels,
        val_labels,
        test_labels,
        train_mask,
        val_mask,
        test_mask,
    ) = load_data()

    g = graph
    cuda = False
    ctx = mx.cpu(0)
    hidden_sizes = [16]
    labels = torch.cat((train_labels, val_labels), 0)
    in_feats = features.shape[1]
    n_edges = graph.number_of_edges()

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    # create APPNP model
    model = APPNP(
        g,
        in_feats,
        hidden_sizes,
        n_classes,
        nd.relu,
        .5,
        .5,
        .1,
        10,
    )

    model.initialize(ctx=ctx)
    n_train_samples = len(train_mask)
    loss_fcn = gluon.loss.SoftmaxCELoss()

    # use optimizer
    print(model.collect_params())
    trainer = gluon.Trainer(
        model.collect_params(),
        "adam",
        {"learning_rate": 1e-2, "wd": 5e-4},
    )

    # initialize graph
    dur = []
    for epoch in range(300):
        if epoch >= 3:
            t0 = time.time()
        # forward
        with mx.autograd.record():
            pred = model(features)
            loss = loss_fcn(pred, labels, mx.nd.expand_dims(train_mask, 1))
            loss = loss.sum() / n_train_samples

        loss.backward()
        trainer.step(batch_size=1)

        if epoch >= 3:
            dur.append(time.time() - t0)
            acc = evaluate(model, features, labels, val_mask)

    # test set accuracy
    acc = evaluate(model, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))


if __name__ == "__main__":
    main()