import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from tqdm import tqdm
import time

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

def infer(g, masks, model):
    # define train/val samples, loss function and optimizer
    # train_nids = torch.nonzero(train_mask).squeeze().type(torch.int32)
    nids = torch.arange(g.num_nodes()).type(torch.int32)

    requests = 0

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    train_dataloader = dgl.dataloading.DataLoader(
        g,
        nids,
        sampler,
        device='cuda',
        batch_size=256,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        # use_prefetch_thread=False,
        # use_alternate_streams=False,
        # pin_prefetcher=False,
    )

    feature_move_time = 0
    # prediction loop
    with torch.no_grad():
        start = time.time()
        for epoch in tqdm(range(100)):
            for input_nodes, output_nodes, mfgs in train_dataloader:
                feat_move_start = time.time()
                inputs = mfgs[0].srcdata["feat"]
                feature_move_time += time.time() - feat_move_start

                model.eval()
                logits = model(mfgs, inputs)
                requests += 1

        elapsed = time.time() - start
        print(f"Elapsed: {elapsed:.2f} seconds, feature movement: {feature_move_time} seconds, requests: {requests}, requests/sec: {requests/elapsed:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="citeseer",
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

    # model inference
    infer(g, masks, model)
