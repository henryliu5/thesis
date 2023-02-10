import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn

class GCN(nn.Module):
    def __init__(self, in_size, out_size, hid_size = 16):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, activation=F.relu, allow_zero_in_degree=True)
        )
        self.layers.append(dglnn.GraphConv(hid_size, out_size, allow_zero_in_degree=True))
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
