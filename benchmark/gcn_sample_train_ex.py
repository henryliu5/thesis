import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset


class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, activation=F.relu)
        )
        self.layers.append(dglnn.GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, mfgs, x):
        h_dst = x[: mfgs[0].num_dst_nodes()]  # <---
        h = self.layers[0](mfgs[0], (x, h_dst))  # <---
        h = F.relu(h)
        h_dst = h[: mfgs[1].num_dst_nodes()]  # <---
        h = self.layers[1](mfgs[1], (h, h_dst))  # <---
        return h
    

    def full_forward(self, g, features):
        g = g.to('cuda')
        h = features.to('cuda')
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h


def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model.full_forward(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        labels = labels.to('cuda')
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(g, features, labels, masks, model):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]

    train_nids = torch.nonzero(train_mask).squeeze().type(torch.int32)

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    train_dataloader = dgl.dataloading.DataLoader(
        g,
        train_nids,
        sampler,
        device='cuda',
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # training loop
    for epoch in range(10):
        for input_nodes, output_nodes, mfgs in train_dataloader:
            inputs = mfgs[0].srcdata["feat"]
            sampled_labels = mfgs[-1].dstdata["label"]
            
            model.train()
            logits = model(mfgs, inputs)
            loss = loss_fcn(logits, sampled_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = evaluate(g, features, labels, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )


if __name__ == "__main__":
    print('This example does sampling of features on GPU\n')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name ('cora', 'citeseer', 'pubmed').",
    )
    args = parser.parse_args()
    print(f"Training with DGL built-in GraphConv module.")

    # load and preprocess dataset
    transform = (
        AddSelfLoop()
    )  # by default, it will first remove self-loops to prevent duplication
    if args.dataset == "cora":
        data = CoraGraphDataset(transform=transform)
    elif args.dataset == "citeseer":
        data = CiteseerGraphDataset(transform=transform)
    elif args.dataset == "pubmed":
        data = PubmedGraphDataset(transform=transform)
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
    g = data[0]

    # Graphs start on HOST
    g = g.int().to("cpu")
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]

    # create GCN model
    in_size = features.shape[1]
    out_size = data.num_classes
    # Model goes on DEVICE
    model = GCN(in_size, 16, out_size).to("cuda")

    # model training
    print("Training...")
    train(g, features, labels, masks, model)

    # test the model
    print("Testing...")
    acc = evaluate(g, features, labels, masks[2], model)
    print("Test accuracy {:.4f}".format(acc))