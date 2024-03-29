from fast_inference.feat_server import FeatureServer, ManagedCacheServer, CountingFeatServer
from fast_inference.device_cache import DeviceFeatureCache
import dgl
import torch
from tqdm import tqdm

def test_feat_server():
    ''' Test feature cache server on small input '''
    device = torch.device('cuda', 0)

    g = dgl.graph(([0, 0, 0, 0, 0], [1, 2, 3, 4, 5]), num_nodes=6)
    x_feats = torch.randn(6, 3)
    y_feats = torch.randn(6, 5, 4) # Multidimensional
    g.ndata['x'] = x_feats
    g.ndata['y'] = y_feats

    cache = DeviceFeatureCache.initialize_cache(init_nids=torch.tensor([0, 2, 4], dtype=torch.long, device=device), num_nodes=g.num_nodes(), feats=g.ndata, device=device, cache_id=0, total_caches=1)

    assert (g.device == torch.device('cpu'))
    # TODO add support for multidimensional features ('y')
    server = FeatureServer([cache], g.num_nodes(), g.ndata, device, 0, 0, track_features=['x'])

    # Set the cache to be nodes 0, 2, 4
    # server.set_static_cache(node_ids=torch.tensor([0, 2, 4], dtype=torch.long, device=device), feats=['x'])
    # Check cache properties
    assert (server.caches[0].cache['x'].shape == torch.Size([3, 3]))
    # assert (server.cache['y'].shape == torch.Size([3, 5, 4]))

    feats, _ = server.get_features(torch.tensor([0, 1, 2], device=device, dtype=torch.long), feats=['x'], mfgs=None)
    
    # Check feats are all on device
    assert(feats['x'].device == torch.device(device))
    # assert(feats['y'].device == torch.device(device))

    # Check values are all equal
    assert (torch.all(torch.eq(feats['x'].cpu(), x_feats[torch.LongTensor([0, 1, 2])])))
    # assert (torch.all(torch.eq(feats['y'].cpu(), y_feats[torch.LongTensor([0, 1, 2])])))

def test_multi_process_feat_server():
    pass

def test_new_server_correctness():
    ''' Check that feature stores actually return the correct value.
        Short dataloading tests race conditions and general correctness.
    '''
    device = torch.device('cuda', 0)

    n = 1000
    g = dgl.graph((torch.zeros(n, dtype=torch.long), torch.arange(n, dtype=torch.long)), num_nodes=n)
    x_feats = torch.arange(n, dtype=torch.float).reshape(n, 1)
    g.ndata['x'] = x_feats

    from fast_inference_cpp import shm_setup
    shm_setup(1, 1)

    cache = DeviceFeatureCache.initialize_cache(init_nids=torch.arange(int(n * 0.80), dtype=torch.long, device=device), num_nodes=g.num_nodes(), feats=g.ndata, device=device, cache_id=0, total_caches=1)
    feat_server = ManagedCacheServer([cache], g.num_nodes(), g.ndata, device, 0, 0, track_features=['x'], executors_per_store=1, total_stores=1)

    feat_server.init_counts(n)
    feat_server.start_manager()

    do_topk = 10
    for i in tqdm(range(5000)):
        requested = torch.randint(0, n, (32,), device=device).unique()

        if i % do_topk == 0:
            feat_server.update_cache()
        
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