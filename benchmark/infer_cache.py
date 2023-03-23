from fast_inference.dataset import InferenceDataset, FastEdgeRepr
from fast_inference.models.factory import load_model
from fast_inference.timer import enable_timers, Timer, print_timer_info, export_timer_info, clear_timers
from fast_inference.feat_server import FeatureServer, CountingFeatServer, LFUServer, ManagedCacheServer
from fast_inference.sampler import InferenceSampler
import dgl
import torch
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
import gc
import argparse
from contextlib import nullcontext
import os
import time

device = torch.device('cuda', 0)

# TODO figure out how to enable inference mode and still make cpp cache server work
@torch.inference_mode()
def main(name, model_name, batch_size, cache_type, subgraph_bias, cache_percent, dir = None, use_gpu_sampling = False, use_pinned_mem = True, MAX_ITERS=5_000_000, run_profiling=False, trials=1):
    BATCH_SIZE = batch_size
    enable_timers()

    infer_percent = 0.1
    if name == 'reddit' or name == 'ogbn-arxiv':
        infer_percent = 0.4
    elif name == 'cora':
        infer_percent = 0.7
    elif name == 'ogbn-papers100M':
        infer_percent = 0.05
        cache_percent /= 4

        if subgraph_bias is not None:
            print("Subgraph bias for ogbn-papers100M not supported")
            return

    infer_data = InferenceDataset(name, infer_percent, partitions=5, force_reload=False, verbose=True)

    g = infer_data[0]
    in_size = g.ndata["feat"].shape[1]
    out_size = infer_data.num_classes

    # Model goes on DEVICE
    model = load_model(model_name, in_size, out_size).to(device)
    model.eval()

    # Convert our graph into just a "logical" graph, all features live in the feature server
    logical_g = dgl.graph(g.edges())

    print(logical_g)
    if name == 'ogbn-papers100M':
        trace = infer_data.create_inference_trace(trace_len=1_000_000, subgraph_bias=subgraph_bias)
    else:
        trace = infer_data.create_inference_trace(subgraph_bias=subgraph_bias)
    n = len(trace)
    # if infer_data._orig_name == 'reddit':
    #     n = len(trace) // 10

    if use_gpu_sampling:
        if name == 'ogbn-papers100M':
            logical_g.create_formats_()
            logical_g.pin_memory_()
            print('Testing ogbn-papers100M, pinning complete')
        else:
            logical_g = logical_g.to(device)

    # TODO remove this and just generate traces with fast mode
    s = time.time()
    in_edge_count = torch.tensor([edge["in"].shape[0] for edge in trace.edges])
    in_edge_endpoints = torch.empty(in_edge_count.sum(), dtype=torch.int64, pin_memory=True)
    torch.cat([edge["in"] for edge in trace.edges], out=in_edge_endpoints)

    out_edge_count = torch.tensor([edge["in"].shape[0] for edge in trace.edges])
    out_edge_endpoints = torch.empty(out_edge_count.sum(), dtype=torch.int64, pin_memory=True)
    torch.cat([edge["in"] for edge in trace.edges], out=out_edge_endpoints)
    print('Creating fast edges done in', time.time() - s)
    trace.edges = FastEdgeRepr(in_edge_endpoints, in_edge_count, out_edge_endpoints, out_edge_count)

    for trial in range(trials):
        clear_timers()
        # Set up feature server
        cache_type = cache_type or 'baseline'
        if cache_type == 'static':
            feat_server = FeatureServer(g.num_nodes(), g.ndata, torch.device('cuda', 0), ['feat'], use_pinned_mem=use_pinned_mem, profile_hit_rate=True)
        elif cache_type == 'count':
            feat_server = CountingFeatServer(g.num_nodes(), g.ndata, torch.device('cuda', 0), ['feat'], use_pinned_mem=use_pinned_mem, profile_hit_rate=True)
        elif cache_type == 'lfu':
            feat_server = LFUServer(g.num_nodes(), g.ndata, torch.device('cuda', 0), ['feat'], use_pinned_mem=use_pinned_mem, profile_hit_rate=True)
        elif cache_type == 'hybrid' or cache_type == 'async':
            feat_server = HybridServer(g.num_nodes(), g.ndata, torch.device('cuda', 0), ['feat'], use_pinned_mem=use_pinned_mem, profile_hit_rate=True)
        elif cache_type == 'baseline':
            feat_server = None
        elif cache_type == 'cpp':
            feat_server = ManagedCacheServer(g.num_nodes(), g.ndata, torch.device('cuda', 0), ['feat'], use_pinned_mem=use_pinned_mem, profile_hit_rate=True)
        else:
            print('Cache type', cache_type, 'not supported')
            exit()
        # # #!! Use only from partition 1
        # part_mapping = infer_data._orig_nid_partitions
        # indices = torch.arange(g.num_nodes())[part_mapping == 2]

        if feat_server:
            # Let's use top 20% of node features for static cache
            out_deg = g.out_degrees()
            _, indices = torch.topk(out_deg, int(g.num_nodes() * cache_percent), sorted=True)
            del out_deg
            feat_server.set_static_cache(indices, ['feat'])

            if cache_type == 'cpp':
                feat_server.start_manager()

            k = 2000
            if name == 'ogbn-papers100M':
                k = 40000

            processed = 0

            print('Caching', indices.shape[0], 'nodes')
            del indices
            gc.collect()
        
            # Need to do this BEFORE converting to logical graph since nodes will be removed
            feat_server.init_counts(g.num_nodes())
        elif use_pinned_mem:
            pin_buf = torch.empty((150_000, g.ndata['feat'].shape[1]), dtype=torch.float, pin_memory=True)

        sampler = InferenceSampler(logical_g)

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) if run_profiling else nullcontext() as prof:
            for i in tqdm(range(0, min(n, MAX_ITERS * BATCH_SIZE), BATCH_SIZE)):
                if i + BATCH_SIZE >= n:
                    continue

                with Timer(name='total'):
                    # TODO make MFG setup work with any batch size and number of layers
                    # TODO see if this MFG setup can be done faster
                    # TODO see GW FastToBlock https://github.com/gwsshs22/dgl/blob/infer-main/src/inference/graph_api.cc
                    mfgs = sampler.sample(trace.nids[i:i+BATCH_SIZE], trace.edges.get_batch(i, i+BATCH_SIZE), use_gpu_sampling=use_gpu_sampling)

                    with Timer(name="dataloading", track_cuda=True):
                        required_feats = mfgs[0].ndata['_ID']['_N']
                        if feat_server:
                            # Cache: (update) + feature gather + CPU-GPU copy
                            with Timer(name="update cache", track_cuda=True):
                                if (i // k) > processed + 1:
                                    if cache_type == 'cpp' or cache_type == 'count':
                                        feat_server.compute_topk()

                                    feat_server.update_cache(['feat'])
                                    processed = i // k  

                            inputs, mfgs = feat_server.get_features(required_feats, feats=['feat'], mfgs=mfgs)
                            inputs = inputs['feat']
                        else:
                            required_feats = required_feats.cpu()
                            if use_pinned_mem:
                                with Timer('resize pin buf'):
                                    pin_buf = pin_buf.resize_((required_feats.shape[0], pin_buf.shape[1]))
                                inputs = pin_buf.narrow(0, 0, required_feats.shape[0])

                            # Baseline: feature gather + CPU-GPU copy
                            with Timer('feature gather'):
                                if use_pinned_mem:
                                    torch.index_select(g.ndata['feat'], 0, required_feats, out=inputs)
                                else:
                                    inputs = g.ndata['feat'][required_feats]

                            with Timer(name="CPU-GPU copy", track_cuda=True):
                                mfgs[0] = mfgs[0].to(device)
                                mfgs[1] = mfgs[1].to(device)
                                inputs = inputs.to(device)
                        

                    with Timer(name='model', track_cuda=True):
                        with Timer('actual forward'):
                            x = model(mfgs, inputs)

                        # Force sync
                        x.cpu()

        if run_profiling:
            prof.export_chrome_trace(f"trace_{model_name}_{name}_{batch_size}_{cache_type}{'_bias_0.8' if subgraph_bias is not None else ''}{'_pinned' if use_pinned_mem else ''}.json")
        print_timer_info()
        if dir != None:
            trial_dir = os.path.join(dir, 'gpu' if use_gpu_sampling else 'cpu')

            if use_pinned_mem:
                trial_dir = os.path.join(trial_dir, 'pinned')

            trial_dir = os.path.join(trial_dir, f'bias_{subgraph_bias}' if subgraph_bias is not None else 'uniform')

            if cache_type == 'baseline':
                trial_dir = os.path.join(trial_dir, cache_type)
            else:
                trial_dir = os.path.join(trial_dir, f'{cache_type}_{cache_percent}')

            export_timer_info(f'{trial_dir}/{model_name.upper()}', {'name': name, 'batch_size': batch_size, 'trial': trial})
            if feat_server:
                feat_server.export_profile(f'{trial_dir}/{model_name.upper()}_cache_info', {'name': name, 'batch_size': batch_size, 'trial': trial})

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Benchmark inference system performance")
    # parser.add_argument('--dataset', type=str, default='ogbn-products',
                        #    help='datasets: reddit, cora, ogbn-products, ogbn-papers100M')
    parser.add_argument('-c', '--cache', type=str,
                           help='Set caching method: static, counting, lfu, hybrid')
    parser.add_argument('-b', '--subgraph_bias', type=float,
                        help='TODO')
    # parser.add_argument('--use_gpu_sampling', action='store_true',
    #                     help='Enable gpu sampling')
    parser.add_argument('--use_pinned_mem', action='store_true',
                           help='Enable cache pinned memory optimization')
    parser.add_argument('--profile', action='store_true',
                           help='Use PyTorch profiler')
    parser.add_argument('-p', '--cache_percent', type=float, default=0.2,
                           help="Cache size, represented as a percentage of the overall graph's nodes")
    
    parser.add_argument('-o', '--output_path', type=str, default='benchmark/data_cache_10/new_cache',
                           help='Output path for timing results')
    
    parser.add_argument('-t', '--trials', type=int, default=1,
                           help="Number of trials to run")
    args = parser.parse_args()

    # main('cora', 'gcn', 256, True)#, dir='benchmark/data/new_index_select')
    models = ['gcn']#, 'sage', 'gat']
    names = ['reddit', 'cora', 'ogbn-products', 'ogbn-papers100M']
    batch_sizes = [1, 64, 128, 256]

    use_gpu_sampling = True
    if use_gpu_sampling:
        names = ['reddit', 'cora', 'ogbn-products']
        names = ['ogbn-papers100M']
        names = ['reddit', 'cora', 'ogbn-products', 'ogbn-papers100M']
        # names = ['ogbn-papers100M']
        names = ['ogbn-products']
        # batch_sizes = [32, 64, 128, 256, 512]
        batch_sizes = [256]
    else:
        # names = ['ogbn-products', 'ogbn-papers100M']
        names = ['ogbn-papers100M']
        # names = ['reddit']

    for model in models:
        for name in names:
            for batch_size in batch_sizes:
                if args.profile:
                    main(name=name, 
                        model_name=model, 
                        batch_size=batch_size, 
                        cache_type=args.cache, 
                        subgraph_bias=args.subgraph_bias,
                        cache_percent=args.cache_percent,
                        dir=None, 
                        use_gpu_sampling=use_gpu_sampling,
                        use_pinned_mem=args.use_pinned_mem,
                        MAX_ITERS=2000,
                        run_profiling=True,
                        trials=1)
                else:
                    main(name=name, 
                        model_name=model, 
                        batch_size=batch_size, 
                        cache_type=args.cache, 
                        subgraph_bias=args.subgraph_bias,
                        cache_percent=args.cache_percent,
                        dir=args.output_path, 
                        use_gpu_sampling=use_gpu_sampling,
                        use_pinned_mem=args.use_pinned_mem,
                        trials=args.trials)
                # main(name=name, model_name=model, batch_size=batch_size, dir='benchmark/data/new_cache_gpu_hybrid', use_gpu_sampling=use_gpu_sampling)
                gc.collect()
                gc.collect()
                gc.collect()
                import time
                time.sleep(5)