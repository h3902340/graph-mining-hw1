from argparse import ArgumentParser
import math


from data_loader import load_data

import torch
import torch.nn as nn
import torch.nn.functional as F


# from model import YourGNNModel # Build your model in model.py

import warnings

from sage import SAGE

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

epsilon = 1 - math.log(2)

def compute_loss(logits, labels):
    y = F.cross_entropy(logits, labels, reduction="none")
    y = torch.log(epsilon + y) - math.log(epsilon)
    return torch.mean(y)

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
        loss = compute_loss(logits[train_mask + val_mask], torch.cat((train_labels, val_labels), 0))
        #loss = compute_loss(logits[train_mask], train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = evaluate(g, features, val_labels, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )

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
    # try to address the sparse feature issue by including the zero features as extended feature
    features = torch.cat((features, features == 0), 1)

    one_hot = torch.zeros([features.shape[0], num_classes], dtype=torch.int, device=device)
    for i in range(60):
        one_hot[i][train_labels[i]] = 1
    for i in range(60, 90):
        one_hot[i][val_labels[i - 60]] = 1
        
    features = torch.cat((features, one_hot), 1)

    # Initialize the model (Baseline Model: GCN)
    in_size = features.shape[1]
    out_size = num_classes

    model_count = 1
    indices_matrix = torch.zeros(
        [model_count, features.shape[0]], dtype=torch.int, device=device
    )
    # labels_merged = torch.cat((train_labels, val_labels), 0)
    for i in range(model_count):
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
        model_sage.eval()
        with torch.no_grad():
            logits = model_sage(graph, features)
            _, indices = torch.max(logits, dim=1)
            indices_matrix[i, :] = indices

        acc = evaluate(graph, features, val_labels, val_mask, model_sage)
        print("SAGE accuracy {:.4f}".format(acc))
    indices_mode = torch.mode(indices_matrix, 0).values

    # Export predictions as csv file
    print("Export predictions as csv file.")
    test_pred_labels = indices_mode[test_mask]
    with open("output.csv", "w") as f:
        f.write("ID,Predict\n")
        for idx, pred in enumerate(test_pred_labels):
            f.write(f"{idx},{int(pred)}\n")