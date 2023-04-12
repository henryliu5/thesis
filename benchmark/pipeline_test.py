''' Benchmark for multiprocessing '''
from fast_inference.dataset import InferenceDataset, FastEdgeRepr
from fast_inference.models.factory import load_model
from fast_inference.feat_server import FeatureServer, CountingFeatServer, LFUServer, ManagedCacheServer
from fast_inference.util import create_feature_stores
from fast_inference.inference_engine import InferenceEngine
from fast_inference.timer import Timer, enable_timers, clear_timers, print_timer_info, export_dict_as_pd
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

class PipelinedDataloader(Process):
    def __init__(self, feature_store, device, num_nodes, cache_type, num_stores, num_executors_per_store, start_barrier):
        super().__init__()
        self.feature_store = feature_store
        self.device = device
        self.num_nodes = num_nodes

        self.cache_type = cache_type
        self.num_stores = num_stores
        self.num_executors_per_store = num_executors_per_store
        self.start_barrier = start_barrier

    def run(self):
        print('device id', self.feature_store.device_index, 'exec id', self.feature_store.executor_id)
        self.feature_store.start_manager()

        # Need to re-pin the feature store buffer
        for k, v in self.feature_store.pinned_buf_dict.items():
            self.feature_store.pinned_buf_dict[k] = v.pin_memory()

        use_prof = False
        enable_timers()

        self.start_barrier.wait()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) if use_prof else nullcontext() as prof:
            for i in range(500):
                required_nids = torch.randint(0, self.num_nodes, (300_000,), device=self.device).unique()

                if i % 10 == 0:
                    self.feature_store.update_cache()

                self.feature_store.get_features(required_nids, ['feat'], None)
        
        if use_prof:
            prof.export_chrome_trace(f'pipeline_trace_{self.device.index * self.feature_store.executor_id}.json')
        print('Lock conflicts', self.feature_store.lock_conflicts)
        export_dict_as_pd(self.feature_store.lock_conflict_trace, 'pipeline_conflicts', {'cache_type': self.cache_type, 
                                                                                         'store_id': self.feature_store.store_id, 
                                                                                         'executor_id': self.feature_store.executor_id,
                                                                                         'num_stores': self.num_stores,
                                                                                         'executors_per_store': self.num_executors_per_store}, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Lock conflict benchmark")
    parser.add_argument('-c', '--cache', type=str, default='cpp_lock',
                           help='Set caching method: static, counting, lfu, hybrid')
    parser.add_argument('-s', '--num_stores', type=int, default=2,
                        help='TODO')
    parser.add_argument('-e', '--executors_per_store', type=int, default=2,
                           help='TODO')

    args = parser.parse_args()

    cache_type = args.cache
    num_stores = args.num_stores
    num_executors_per_store = args.executors_per_store

    num_nodes = 2_000_000
    g = dgl.graph((torch.zeros(num_nodes, dtype=torch.long), torch.arange(num_nodes, dtype=torch.long)), num_nodes=num_nodes)
    x_feats = torch.randn(num_nodes, 100)
    g.ndata['feat'] = x_feats

    feature_stores = create_feature_stores(cache_type, num_stores, num_executors_per_store, g, ['feat'], 0.2, True, profile_hit_rate=True, pinned_buf_size=1_000_000)

    start_barrier = Barrier(num_stores * num_executors_per_store)

    dl_processes = []
    for i in range(num_stores):
        for j in range(num_executors_per_store):
            dl_processes.append(PipelinedDataloader(feature_stores[i][j], torch.device('cuda', i), num_nodes, cache_type, num_stores, num_executors_per_store, start_barrier))
    
    s = time.perf_counter()
    [p.start() for p in dl_processes]
    [p.join() for p in dl_processes]
    print(time.perf_counter() - s)