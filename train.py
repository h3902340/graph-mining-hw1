from argparse import ArgumentParser
import math

from data_loader import load_data

import torch
import torch.nn as nn
import torch.nn.functional as F
from gatv2_conv_DGL import GATv2Conv
from gcn import GCN
from grace import Grace
import optuna
import dgl.data
import matplotlib.pyplot as plt
import networkx as nx

from gat import GAT

# from model import YourGNNModel # Build your model in model.py

import os
import warnings

from gatv2 import GATv2
from gcn_three_layer import GCN_THREE
from sage import SAGE
from sage_mean import SAGE_MEAN

warnings.filterwarnings("ignore")


def evaluate(g, features, labels, mask, model):
    """Evaluate model accuracy"""
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def train(
    g,
    features,
    train_labels,
    val_labels,
    train_mask,
    val_mask,
    model,
    epochs,
    es_iters=None,
):

    # define train/val samples, loss function and optimizer
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # If early stopping criteria, initialize relevant parameters
    if es_iters:
        print("Early stopping monitoring on")
        loss_min = 1e8
        es_i = 0

    # training loop
    for epoch in range(epochs):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask + val_mask], torch.cat((train_labels, val_labels), 0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # acc = evaluate(g, features, val_labels, val_mask, model)
        # print(
        #    "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
        #        epoch, loss.item(), acc
        #    )
        # )

        val_loss = loss_fcn(logits[val_mask], val_labels).item()
        if es_iters:
            if val_loss < loss_min:
                loss_min = val_loss
                es_i = 0
            else:
                es_i += 1

            if es_i >= es_iters:
                print(f"Early stopping at epoch={epoch+1}")
                break


if __name__ == "__main__":

    parser = ArgumentParser()
    # you can add your arguments if needed
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--es_iters", type=int, help="num of iters to trigger early stopping"
    )
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    if args.use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Load data
    (
        features,
        graph,
        num_classes,
        train_labels,
        val_labels,
        test_labels,
        train_mask,
        val_mask,
        test_mask,
    ) = load_data()

    train_mask = torch.tensor(train_mask).to(device)
    train_labels = torch.tensor(train_labels).to(device)
    val_mask = torch.tensor(val_mask).to(device)
    val_labels = torch.tensor(val_labels).to(device)
    test_mask = torch.tensor(test_mask).to(device)
    test_labels = torch.tensor(test_labels).to(device)
    graph = graph.to(device)
    features = features.to(device)
    
    '''options = {
    'node_color': 'black',
    'node_size': 20,
    'width': 1,
    }
    G = dgl.to_networkx(graph)
    plt.figure(figsize=[15,7])
    nx.draw(G, **options)'''
    
    one_hot = torch.zeros([features.shape[0], num_classes], dtype=torch.int, device=device)
    for i in range(60):
        one_hot[i][train_labels[i]] = 1
    for i in range(60, 90):
        one_hot[i][val_labels[i - 60]] = 1
        
    features = torch.cat((features, one_hot), 1)
    
    for i in range(60):
        if not train_mask[i]:
            print(f"train_mask wrong! i={i}")
            
    for i in range(60, 90):
        if not val_mask[i]:
            print(f"val_mask wrong! i={i}")

    # Initialize the model (Baseline Model: GCN)
    """TODO: build your own model in model.py and replace GCN() with your model"""
    in_size = features.shape[1]
    out_size = num_classes
    # model = GAT(in_size, 16, out_size, [16, 1]).to(device)
    num_layers = 2
    num_heads = 16
    num_hidden = 16
    num_out_heads = 1
    heads = ([num_heads] * num_layers) + [num_out_heads]
    
    model_sage = SAGE(in_size, 24, out_size)
    train(
        graph,
        features,
        train_labels,
        val_labels,
        train_mask,
        val_mask,
        model_sage,
        args.epochs,
        args.es_iters,
    )
    
    model_gcn = GCN(in_size, 24, out_size)
    train(
        graph,
        features,
        train_labels,
        val_labels,
        train_mask,
        val_mask,
        model_gcn,
        args.epochs,
        args.es_iters,
    )
    
    model_gat = GAT(in_size, 24, out_size, [2, 2])
    train(
        graph,
        features,
        train_labels,
        val_labels,
        train_mask,
        val_mask,
        model_gat,
        args.epochs,
        args.es_iters,
    )
    
    '''def objective(trial):
        hid_size = trial.suggest_int('hid_size', 4, 32)
        #dropout = trial.suggest_float(f'dropout', 0, 1)
        #aggregator_type = trial.suggest_int(f'aggregator_type', 0, 3)
        model = SAGE(in_size, hid_size, out_size, .3, 2, 0)

        # model training
        #print("Training...")
        train(
            graph,
            features,
            train_labels,
            val_labels,
            train_mask,
            val_mask,
            model,
            args.epochs,
            args.es_iters,
        )

        #print("Testing...")
        acc = evaluate(graph, features, val_labels, val_mask, model)
        print("Test accuracy {:.4f}".format(acc))
        return acc
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)'''


    print("Testing...")
    acc = evaluate(graph, features, val_labels, val_mask, model_sage)
    print("SAGE accuracy {:.4f}".format(acc))
    acc = evaluate(graph, features, val_labels, val_mask, model_gcn)
    print("GCN accuracy {:.4f}".format(acc))
    acc = evaluate(graph, features, val_labels, val_mask, model_gat)
    print("GAT accuracy {:.4f}".format(acc))
        
    model_sage.eval()
    with torch.no_grad():
        logits = model_sage(graph, features)
        logits = logits[test_mask]
        _, indices_sage = torch.max(logits, dim=1)
        
    model_gcn.eval()
    with torch.no_grad():
        logits = model_gcn(graph, features)
        logits = logits[test_mask]
        _, indices_gcn = torch.max(logits, dim=1)
        
    model_gat.eval()
    with torch.no_grad():
        logits = model_gat(graph, features)
        logits = logits[test_mask]
        _, indices_gat = torch.max(logits, dim=1)

    # Export predictions as csv file
    print("Export predictions as csv file.")
    with open("output.csv", "w") as f:
        f.write("ID,Predict\n")
        all_different = 0
        one_different = 0
        for idx, pred in enumerate(indices_sage):
            if pred != indices_gcn[idx] and pred != indices_gat[idx] and indices_gcn[idx] != indices_gat[idx]:
                all_different = all_different + 1
            if pred != indices_gcn[idx] and pred == indices_gat[idx]:
                one_different = one_different + 1
            if pred != indices_gat[idx] and pred == indices_gcn[idx]:
                one_different = one_different + 1
            if indices_gcn[idx] != pred and indices_gcn[idx] == indices_gat[idx]:
                one_different = one_different + 1
            pred_mix = torch.round((pred * .79 + indices_gcn[idx] * .76 + indices_gat[idx] * .76)/(.79 + .76 + .76))
            f.write(f"{idx},{int(pred)}\n")
        print(f"all_different: {all_different}")
        print(f"one_different: {one_different}")
    # Please remember to upload your output.csv file to Kaggle for scoring
