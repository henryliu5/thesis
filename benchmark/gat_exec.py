from ogb.nodeproppred import DglNodePropPredDataset
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch as th
import dgl.function as fn
from dgl.ops.edge_softmax import edge_softmax
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils.internal import expand_as_pair

import dgl

from fast_inference.timer import Timer, enable_timers, print_timer_info
from tqdm import tqdm

class GAT(nn.Module):
    def __init__(self, in_size, out_size, hid_size=16, heads=None):
        if heads == None:
            heads = [8, 8]
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GAT
        self.layers.append(GATConv(in_size, 256, 4, feat_drop=0.6, attn_drop=0.6, allow_zero_in_degree=True, heads_aggregation='flatten', activation=F.elu))
        self.layers.append(GATConv(1024, 256, 4, feat_drop=0.6, attn_drop=0.6, allow_zero_in_degree=True, heads_aggregation='flatten', activation=F.elu))
        self.layers.append(GATConv(1024, out_size, 6, feat_drop=0.6, attn_drop=0.6, allow_zero_in_degree=True, heads_aggregation='mean', activation=None))

    def forward(self, blocks, h):
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)

        return h

# pylint: enable=W0235
class GATConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 heads_aggregation=None, # 'flatten' or 'mean
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)

        self.reset_parameters()
        self.heads_aggregation = heads_aggregation
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)

            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))

            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))

            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']

            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
            
            if self.heads_aggregation == "flatten":
                rst = rst.flatten(1)
            elif self.heads_aggregation == "mean":
                rst = rst.mean(1)

            # activation
            if self.activation:
                rst = self.activation(rst)

            return rst

def train(g, features, labels, masks, model):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]

    train_nids = torch.nonzero(train_mask).squeeze().type(torch.int32)

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
    train_dataloader = dgl.dataloading.DataLoader(
        g,
        train_nids,
        sampler,
        device='cpu',
        batch_size=256,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    i = 0
    # training loop
    for epoch in range(1000):
        for input_nodes, output_nodes, mfgs in tqdm(train_dataloader):
            with Timer('CPU-GPU copy', track_cuda=True):
                inputs = mfgs[0].srcdata["feat"].to('cuda')
                sampled_labels = mfgs[-1].dstdata["label"].to('cuda')
                mfgs = [mfg.to('cuda') for mfg in mfgs]
            
            model.train()
            with Timer('model forward', track_cuda=True):
                logits = model(mfgs, inputs)    
            with Timer('backprop + optim', track_cuda=True):
                loss = loss_fcn(logits, sampled_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if i > 5:
                print_timer_info(ignore_first_n=5)
            i += 1

        # acc = evaluate(g, features, labels, val_mask, model)
        acc = -1
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )


if __name__ == "__main__":
    enable_timers()

    data = DglNodePropPredDataset('ogbn-products')
    print(data[0])
    g, labels = data[0]

    # Graphs start on HOST
    g = g.int().to("cpu")
    features = g.ndata["feat"]
    # data.get_idx_split()
    # masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]
    # Just use everything
    masks = torch.ones(g.num_nodes(), dtype=torch.bool), torch.ones(g.num_nodes(), dtype=torch.bool), torch.ones(g.num_nodes(), dtype=torch.bool)
    g.ndata['label'] = labels[:, 0]
    g = dgl.add_self_loop(g)

    # create GCN model
    in_size = features.shape[1]
    out_size = data.num_classes
    # Model goes on DEVICE
    # model = GCN(in_size, 16, out_size).to("cuda")
    model = GAT(in_size, out_size).to('cuda')

    # model training
    print("Training...")
    train(g, features, labels, masks, model)

    print_timer_info()
    # # test the model
    # print("Testing...")
    # acc = evaluate(g, features, labels, masks[2], model)
    # print("Test accuracy {:.4f}".format(acc))
