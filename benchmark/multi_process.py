''' Benchmark for multiprocessing '''
from fast_inference.dataset import InferenceDataset, FastEdgeRepr
from fast_inference.models.factory import load_model
from fast_inference.feat_server import FeatureServer, CountingFeatServer, LFUServer, ManagedCacheServer
from fast_inference.util import create_feature_stores
from fast_inference.inference_engine import InferenceEngine
from fast_inference.request_generator import RequestGenerator, ResponseRecipient
import dgl
import torch
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
import time

from torch.multiprocessing import Queue, Process, Barrier, set_start_method, Lock
try:
    set_start_method('spawn')
except RuntimeError:
    pass

import gc
import argparse
from contextlib import nullcontext
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Benchmark inference system performance")
    # parser.add_argument('--dataset', type=str, default='ogbn-products',
                        #    help='datasets: reddit, cora, ogbn-products, ogbn-papers100M')
    parser.add_argument('-c', '--cache', type=str,
                           help='Set caching method: static, counting, lfu, hybrid')
    parser.add_argument('-b', '--subgraph_bias', type=float, default=None,
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

    num_engines = torch.cuda.device_count()
    dataset = 'ogbn-products'
    batch_size = 256
    max_iters = 512000
    infer_percent = 0.1
    model_name = 'gcn'
    subgraph_bias = args.subgraph_bias
    cache_percent = 0.2
    dir = 'multi_testing'
    use_gpu_sampling = True
    use_pinned_mem = True
    cache_type = args.cache
    num_trials = 3

    trial_dir = os.path.join(dir, 'gpu' if use_gpu_sampling else 'cpu')

    if use_pinned_mem:
        trial_dir = os.path.join(trial_dir, 'pinned')

    trial_dir = os.path.join(trial_dir, f'bias_{subgraph_bias}' if subgraph_bias is not None else 'uniform')

    if cache_type == 'baseline':
        trial_dir = os.path.join(trial_dir, cache_type)
    else:
        trial_dir = os.path.join(trial_dir, f'{cache_type}_{cache_percent}')

    infer_data = InferenceDataset(
        dataset, infer_percent, partitions=5, force_reload=False, verbose=True)
    trace = infer_data.create_inference_trace(subgraph_bias=subgraph_bias)

    # s = time.time()
    # in_edge_count = torch.tensor([edge["in"].shape[0] for edge in trace.edges])
    # in_edge_endpoints = torch.empty(in_edge_count.sum(), dtype=torch.int64, pin_memory=True)
    # torch.cat([edge["in"] for edge in trace.edges], out=in_edge_endpoints)

    # out_edge_count = torch.tensor([edge["in"].shape[0] for edge in trace.edges])
    # out_edge_endpoints = torch.empty(out_edge_count.sum(), dtype=torch.int64, pin_memory=True)
    # torch.cat([edge["in"] for edge in trace.edges], out=out_edge_endpoints)
    # trace.edges = FastEdgeRepr(in_edge_endpoints, in_edge_count, out_edge_endpoints, out_edge_count)
    # print('Creating fast edges done in', time.time() - s)

    request_queue = Queue()
    response_queue = Queue()
    # # Request generator + Inference Engines + Response recipient
    start_barrier = Barrier(2 + num_engines)
    finish_barrier = Barrier(2 + num_engines)
    trial_barriers = [Barrier(2 + num_engines) for _ in range(num_trials)]

    request_generator = RequestGenerator(request_queue=request_queue, start_barrier=start_barrier, finish_barrier=finish_barrier, trial_barriers=trial_barriers,
                                         num_engines=num_engines,
                                         trace=trace, batch_size=batch_size, max_iters=max_iters, rate=30, trials=num_trials)
    request_generator.start()
    response_recipient = ResponseRecipient(response_queue=response_queue, start_barrier=start_barrier, finish_barrier=finish_barrier, trial_barriers=trial_barriers,
                                           num_engines=num_engines,
                                            dataset=dataset, model_name=model_name, batch_size=batch_size, output_path=trial_dir)
    response_recipient.start()

    g = infer_data[0]
    # Model goes on DEVICE
    in_size = g.ndata["feat"].shape[1]
    out_size = infer_data.num_classes
    model = load_model(model_name, in_size, out_size)
    model.eval()

    num_nodes = g.num_nodes()
    feature_stores = create_feature_stores(cache_type, num_engines, g, ['feat'], cache_percent, use_pinned_mem, profile_hit_rate=True, pinned_buf_size=1_000_000)

    logical_g = dgl.graph(g.edges())

    engines = []
    # t = torch.ones(3, device='cuda')
    for device_id in range(num_engines):
        # engines.append(CUDATest(t, device_id))
        # engines[-1].start()
        print('Creating InferenceEngine for device_id:', device_id)
        engine = InferenceEngine(request_queue=request_queue,
                                response_queue=response_queue,
                                start_barrier=start_barrier,
                                finish_barrier=finish_barrier,
                                trial_barriers=trial_barriers,
                                num_engines=num_engines,
                                device=torch.device('cuda', device_id),
                                feature_store=feature_stores[device_id],
                                logical_g = logical_g,
                                model=model,
                                # Benchmarking info
                                dataset=dataset, model_name=model_name, batch_size=batch_size, output_path=trial_dir)
        engine.start()
        engines.append(engine)

    # engines[0].incr()
    # engines[1].decr()

    [engine.join() for engine in engines]
    request_generator.join()
    response_recipient.join()
    print('main exiting')
