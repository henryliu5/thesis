from fast_inference.feat_server import FeatureServer, ManagedCacheServer, CountingFeatServer
import dgl
import torch
from tqdm import tqdm

def test_feat_server():
    ''' Test feature cache server on small input '''
    device = 'cuda:0'

    g = dgl.graph(([0, 0, 0, 0, 0], [1, 2, 3, 4, 5]), num_nodes=6)
    x_feats = torch.randn(6, 3)
    y_feats = torch.randn(6, 5, 4) # Multidimensional
    g.ndata['x'] = x_feats
    g.ndata['y'] = y_feats

    assert (g.device == torch.device('cpu'))
    server = FeatureServer(g.num_nodes(), g.ndata, track_features=['x', 'y'], device=device)

    # Set the cache to be nodes 0, 2, 4
    server.set_static_cache(node_ids=torch.LongTensor([0, 2, 4]), feats=['x', 'y'])
    # Check cache properties
    assert (server.cache['x'].shape == torch.Size([3, 3]))
    assert (server.cache['y'].shape == torch.Size([3, 5, 4]))

    feats, _ = server.get_features(torch.tensor([0, 1, 2], device=device, dtype=torch.long), feats=['x', 'y'], mfgs=None)
    
    # Check feats are all on device
    assert(feats['x'].device == torch.device(device))
    assert(feats['y'].device == torch.device(device))

    # Check values are all equal
    assert (torch.all(torch.eq(feats['x'].cpu(), x_feats[torch.LongTensor([0, 1, 2])])))
    assert (torch.all(torch.eq(feats['y'].cpu(), y_feats[torch.LongTensor([0, 1, 2])])))

def test_new_server_correctness():
    ''' Check that feature stores actually return the correct value.
        Short dataloading tests race conditions and general correctness.
    '''
    device = 'cuda:0'

    n = 1000
    g = dgl.graph((torch.zeros(n, dtype=torch.long), torch.arange(n, dtype=torch.long)), num_nodes=n)
    x_feats = torch.arange(n, dtype=torch.float).reshape(n, 1)
    g.ndata['x'] = x_feats

    feat_server = ManagedCacheServer(g.num_nodes(), g.ndata, device=device, track_features=['x'])
    feat_server.set_static_cache(node_ids=torch.arange(int(n * 0.80), dtype=torch.long), feats=['x'])
    feat_server.init_counts(n)

    do_topk = 10
    for i in tqdm(range(5000)):
        requested = torch.randint(0, n, (32,), device=device).unique()

        if i % do_topk == 0:
            feat_server.compute_topk()
            feat_server.update_cache(['x'])
        
        result, _ = feat_server.get_features(requested, ['x'])

        expected = x_feats[requested.cpu()]
        assert (result['x'].device == torch.device(device))
        
        if not torch.all(torch.eq(result['x'].cpu(), expected)):
            print(result['x'].cpu())
            print(expected)

        assert (torch.all(torch.eq(result['x'].cpu(), expected)))
    

if __name__ == '__main__':
    # from torch.profiler import profile, ProfilerActivity, record_function
    test_new_server_correctness()
    # rand_seed = 12345
    # torch.manual_seed(rand_seed)
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    #     test_new_server_correctness()
    # prof.export_chrome_trace('debug.json')