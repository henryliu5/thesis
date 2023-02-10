from fast_inference.feat_server import FeatureServer
import dgl
import torch

def test_feat_server():
    ''' Test feature cache server on small input '''
    device = 'cuda:0'

    g = dgl.graph(([0, 0, 0, 0, 0], [1, 2, 3, 4, 5]), num_nodes=6)
    x_feats = torch.randn(6, 3)
    y_feats = torch.randn(6, 5, 4) # Multidimensional
    g.ndata['x'] = x_feats
    g.ndata['y'] = y_feats

    assert (g.device == torch.device('cpu'))
    server = FeatureServer(g, device=device)

    # Set the cache to be nodes 0, 2, 4
    server.set_static_cache(node_ids=torch.LongTensor([0, 2, 4]), feats=['x', 'y'])
    # Check cache properties
    assert (server.cache['x'].shape == torch.Size([3, 3]))
    assert (server.cache['y'].shape == torch.Size([3, 5, 4]))

    feats = server.get_features(torch.LongTensor([0, 1, 2]), feats=['x', 'y'])
    
    # Check feats are all on device
    assert(feats['x'].device == torch.device(device))
    assert(feats['y'].device == torch.device(device))

    # Check values are all equal
    assert (torch.all(torch.eq(feats['x'].cpu(), x_feats[torch.LongTensor([0, 1, 2])])))
    assert (torch.all(torch.eq(feats['y'].cpu(), y_feats[torch.LongTensor([0, 1, 2])])))

