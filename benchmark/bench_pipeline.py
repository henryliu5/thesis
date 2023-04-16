''' Benchmark for multiprocessing '''
from fast_inference.dataset import InferenceDataset
from fast_inference.models.factory import load_model
from fast_inference.request_generator import RequestGenerator, ResponseRecipient
from fast_inference.pipeline.pipeline import create_pipeline
import torch

from torch.multiprocessing import Queue, Barrier, set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

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
    
    parser.add_argument('-o', '--output_path', type=str, default='multi_testing',
                           help='Output path for timing results')
    
    parser.add_argument('-t', '--trials', type=int, default=1,
                           help="Number of trials to run")
    parser.add_argument('-e', '--executors_per_store', type=int, default=4,
                            help="Number of executors per feature store")
    parser.add_argument('-g', '--gpus', type=int, default=torch.cuda.device_count(),
                            help="Number of feature stores")
    args = parser.parse_args()

    num_devices = args.gpus
    executors_per_store = args.executors_per_store
    num_engines = num_devices * executors_per_store

    MULTIPLIER = 1
    dataset = 'ogbn-products'
    batch_size = 128
    max_iters = 512000
    infer_percent = 0.1 * MULTIPLIER
    model_name = 'gcn'
    subgraph_bias = args.subgraph_bias
    cache_percent = 0.2
    dir = args.output_path
    use_gpu_sampling = True
    use_pinned_mem = True
    cache_type = args.cache
    num_trials = args.trials

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
    trace = infer_data.create_inference_trace(trace_len=256000 * MULTIPLIER, subgraph_bias=subgraph_bias)


    # Pipeline stages per device
    samplers = 4
    data_loaders = 4
    model_executors = 4

    num_worker_procs = num_devices * (samplers + data_loaders + model_executors)

    request_queue = Queue()
    response_queue = Queue()
    # # Request generator + Inference Engines + Response recipient
    start_barrier = Barrier(2 + num_worker_procs)
    finish_barrier = Barrier(2 + num_worker_procs)
    trial_barriers = [Barrier(5) for _ in range(num_trials)]

    request_generator = RequestGenerator(request_queue=request_queue, start_barrier=start_barrier, finish_barrier=finish_barrier, trial_barriers=trial_barriers,
                                         num_engines=num_engines,
                                         trace=trace, batch_size=batch_size, max_iters=500, rate=0, trials=num_trials)
    request_generator.start()
    response_recipient = ResponseRecipient(response_queue=response_queue, start_barrier=start_barrier, finish_barrier=finish_barrier, trial_barriers=trial_barriers,
                                           num_engines=num_engines, num_devices=num_devices, executors_per_store=executors_per_store,
                                            dataset=dataset, model_name=model_name, batch_size=batch_size, output_path=trial_dir)
    response_recipient.start()

    g = infer_data[0]
    in_size = g.ndata["feat"].shape[1]
    out_size = infer_data.num_classes
    model = load_model(model_name, in_size, out_size)
    model.eval()

    num_nodes = g.num_nodes()

    manager, pipeline_workers = create_pipeline(num_devices=num_devices, 
                                        samplers=samplers,
                                        data_loaders=data_loaders,
                                        model_executors=model_executors,
                                        cache_type=cache_type,
                                        cache_percent=cache_percent,
                                        g=g,
                                        model=model,
                                        request_queue=request_queue,
                                        response_queue=response_queue,
                                        barriers={'start': start_barrier, 'finish': finish_barrier})

    [worker.join() for worker in pipeline_workers]
    request_generator.join()
    response_recipient.join()
    print('main exiting')
