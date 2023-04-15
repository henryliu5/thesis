''' Benchmark for multiprocessing '''
from fast_inference.dataset import InferenceDataset, FastEdgeRepr
from fast_inference.models.factory import load_model
from fast_inference.feat_server import FeatureServer, CountingFeatServer, LFUServer, ManagedCacheServer
from fast_inference.util import create_feature_stores
from fast_inference.inference_engine import InferenceEngine
from fast_inference.timer import Timer, enable_timers, clear_timers, print_timer_info
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
    def __init__(self, feature_store, device, num_nodes, expected_feats):
        super().__init__()
        self.feature_store = feature_store
        self.device = device
        self.num_nodes = num_nodes
        self.expected_feats = expected_feats

    def run(self):
        if type(self.feature_store) == ManagedCacheServer:
            self.feature_store.start_manager()
        if type(self.feature_store) == CountingFeatServer:
            self.feature_store.init_locks()

        # print(self.feature_store.cache['feat'].is_shared())

        # Need to re-pin the feature store buffer
        for k, v in self.feature_store.pinned_buf_dict.items():
            self.feature_store.pinned_buf_dict[k] = v.pin_memory()

        use_prof = False
        enable_timers()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) if use_prof else nullcontext() as prof:
            for i in tqdm(range(1_000), disable=self.device.index != 0 or not self.feature_store.is_leader):
                requested = torch.randint(0, self.num_nodes, (300_000,), device=self.device).unique()

                if i % 10 == 0:
                    self.feature_store.update_cache()

                result, _ = self.feature_store.get_features(requested, ['feat'], None)

                expected = self.expected_feats[requested.cpu()]
                assert (result['feat'].device == torch.device(self.device))
                
                if not torch.all(torch.eq(result['feat'].cpu(), expected)):
                    print('equality check failed')
                    print(result['feat'].cpu())
                    print(expected)

                assert (torch.all(torch.eq(result['feat'].cpu(), expected)))
        
        if use_prof:
            prof.export_chrome_trace(f'pipeline_trace_{self.device.index}.json')
        print('Lock conflicts', self.feature_store.lock_conflicts)


def _check_multiprocess_correctness(policy):
    num_devices = torch.cuda.device_count()
    num_executors_per_store = 2
    num_engines = num_devices * num_executors_per_store

    num_nodes = 2_000_000
    g = dgl.graph((torch.zeros(num_nodes, dtype=torch.long), torch.arange(num_nodes, dtype=torch.long)), num_nodes=num_nodes)
    expected_feats = torch.randn(num_nodes, 5)
    g.ndata['feat'] = expected_feats

    feature_stores = create_feature_stores(policy, num_devices, num_executors_per_store, g, ['feat'], 0.2, True, profile_hit_rate=True, pinned_buf_size=1_000_000)

    dl_processes = []
    for i in range(num_devices):
        for j in range(num_executors_per_store):
            dl_processes.append(PipelinedDataloader(feature_stores[i][j], torch.device('cuda', i), num_nodes, expected_feats))
        
    [p.start() for p in dl_processes]
    [p.join() for p in dl_processes]
    print(dl_processes)
    for p in dl_processes:
        assert(p.exitcode == 0), "Probably a race!"

def test_cpp_race():
    _check_multiprocess_correctness('cpp')

def test_cpp_lock_race():
    _check_multiprocess_correctness('cpp_lock')

def test_count_race():
    _check_multiprocess_correctness('count')

def test_static_race():
    _check_multiprocess_correctness('static')


if __name__ == '__main__':
    # _check_multiprocess_correctness('static')
    # _check_multiprocess_correctness('count')
    _check_multiprocess_correctness('cpp')
    _check_multiprocess_correctness('cpp_lock')