from fast_inference.dataset import InferenceDataset
from fast_inference.models.factory import load_model
from fast_inference.timer import enable_timers, Timer, print_timer_info, export_timer_info, clear_timers
from fast_inference.feat_server import FeatureServer
import dgl
import torch
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
import gc

device = 'cuda'

@torch.no_grad()
def main(name, model_name, batch_size, dir = None, use_gpu_sampling = False):
    BATCH_SIZE = batch_size
    enable_timers()
    clear_timers()
    infer_data = InferenceDataset(name, 0.1, force_reload=False, verbose=True)
    g = infer_data[0]
    in_size = g.ndata["feat"].shape[1]
    out_size = infer_data.num_classes

    # Set up feature server
    feat_server = FeatureServer(g, 'cuda', ['feat'])
    out_deg = g.out_degrees()
    
    # Let's use top 20% of node features for static cache
    _, indices = torch.topk(out_deg, int(g.num_nodes() * 0.2), sorted=False)
    del out_deg
    feat_server.set_static_cache(indices, ['feat'])
    print('Caching', indices.shape[0], 'nodes')
    del indices
    gc.collect()

    # Convert our graph into just a "logical" graph, all features live in the feature server
    g = dgl.graph(g.edges())

    if use_gpu_sampling:
        g = g.to(device)

    # Model goes on DEVICE
    model = load_model(model_name, in_size, out_size).to(device)
    model.eval()

    print(g)
    trace = infer_data.create_inference_trace(subgraph_bias=0.8)
    n = len(trace)
    if infer_data._orig_name == 'reddit':
        n = len(trace) // 10

    MAX_ITERS = 1000
    for i in tqdm(range(0, min(n, MAX_ITERS * BATCH_SIZE), BATCH_SIZE)):
        if i + BATCH_SIZE >= n:
            continue

        # TODO decide what to do if multiple infer requests for same node id
        # orig_new_nid = trace.nids[i:i+BATCH_SIZE]
        new_nid = []
        adj_nids = []
        sizes = []
        s = set()
        for idx in range(i, i + BATCH_SIZE):
            if trace.nids[idx].item() not in s:
                adj_nids.append(trace.edges[idx]["in"])
                sizes.append(trace.edges[idx]["in"].shape[0])
                new_nid.append(trace.nids[idx].item())
                s.add(trace.nids[idx].item())
        
        new_nid = torch.tensor(new_nid)
        # assert(new_nid.shape == orig_new_nid.shape)
        # assert(new_nid.shape == new_nid.unique().shape) 
        if BATCH_SIZE == 1:
            new_nid = new_nid.reshape(1)

        with Timer(name='total'):
            # TODO make MFG setup work with any batch size and number of layers
            # TODO see if this MFG setup can be done faster
            # TODO see GW FastToBlock https://github.com/gwsshs22/dgl/blob/infer-main/src/inference/graph_api.cc

            mfgs = []

            with Timer('sampling', track_cuda=use_gpu_sampling):
                # TODO test this batching very carefully
                # TODO reason to be suspicious: https://github.com/dmlc/dgl/issues/4512
                required_nodes = torch.cat(adj_nids)
                required_nodes_unique = required_nodes.unique()
                interleave_count = torch.tensor(sizes)
                # required_nodes = torch.cat([trace.edges[idx]["in"] for idx in range(i, i+BATCH_SIZE)])
                # interleave_count = torch.tensor([trace.edges[idx]["in"].shape[0] for idx in range(i, i+BATCH_SIZE)])

                # Create first layer message flow graph by looking at required neighbors
                all_seeds = torch.cat((required_nodes_unique, new_nid))

                if use_gpu_sampling:
                    # NOTE roughly 10x faster
                    frontier = dgl.sampling.sample_neighbors(g, required_nodes_unique.to(device), -1)
                else:
                    frontier = dgl.sampling.sample_neighbors(g, required_nodes_unique, -1)

                first_mfg = dgl.to_block(frontier, all_seeds) # Need to do cat here as should have target node

                # Create a message flow graph using the new edges
                mfg = dgl.graph((required_nodes, torch.repeat_interleave(new_nid, interleave_count)))
                last_mfg = dgl.to_block(mfg, new_nid)
            
                mfgs.append(first_mfg)
                mfgs.append(last_mfg)

            with Timer(name="dataloading", track_cuda=True):
                required_feats = mfgs[0].ndata['_ID']['_N']
                # required_feats = torch.randint(0, 111059956, (124364,))
                inputs, mfgs = feat_server.get_features(required_feats, feats=['feat'], mfgs=mfgs)
                inputs = inputs['feat']

            with Timer(name='model', track_cuda=True):
                x = model(mfgs, inputs)
                # Force sync
                x.cpu()

    print_timer_info()
    if dir != None:
        export_timer_info(f'{dir}/{model_name.upper()}', {'name': name, 'batch_size': batch_size})

if __name__ == '__main__':
    # main('cora', 'gcn', 256, True)#, dir='benchmark/data/new_index_select')
    models = ['gcn']#, 'sage', 'gat']
    names = ['reddit', 'cora', 'ogbn-products', 'ogbn-papers100M']
    batch_sizes = [1, 64, 128, 256]

    use_gpu_sampling = True
    if use_gpu_sampling:
        path = 'benchmark/data/new_cache_gpu_bias_0.8'
        names = ['reddit', 'cora', 'ogbn-products']
    else:
        path = 'benchmark/data/new_cache'

    for model in models:
        for name in names:
            for batch_size in batch_sizes:
                main(name=name, model_name=model, batch_size=batch_size, dir=path, use_gpu_sampling=use_gpu_sampling)
                gc.collect()
                gc.collect()
                gc.collect()