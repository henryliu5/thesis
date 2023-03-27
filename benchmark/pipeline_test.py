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
    def __init__(self, feature_store, device, num_nodes):
        super().__init__()
        self.feature_store = feature_store
        self.device = device
        self.num_nodes = num_nodes

    def run(self):
        self.feature_store.start_manager()

        # Need to re-pin the feature store buffer
        for k, v in self.feature_store.pinned_buf_dict.items():
            self.feature_store.pinned_buf_dict[k] = v.pin_memory()

        use_prof = True
        enable_timers()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) if use_prof else nullcontext() as prof:
            for i in range(500):
                required_nids = torch.randint(0, self.num_nodes, (300_000,), device=self.device).unique()

                if i % 10 == 0:
                    self.feature_store.compute_topk()
                    self.feature_store.update_cache(['feat'])

                self.feature_store.get_features(required_nids, ['feat'], None)
        
        if use_prof:
            prof.export_chrome_trace(f'pipeline_trace_{self.device.index}.json')
        print('Lock conflicts', self.feature_store.lock_conflicts)

if __name__ == '__main__':
    num_engines = 2
    device = torch.device('cuda', 0)

    num_nodes = 2_000_000
    g = dgl.graph((torch.zeros(num_nodes, dtype=torch.long), torch.arange(num_nodes, dtype=torch.long)), num_nodes=num_nodes)
    x_feats = torch.randn(num_nodes, 100)
    g.ndata['feat'] = x_feats

    feature_stores = create_feature_stores('cpp_lock', num_engines, g, ['feat'], 0.2, True, profile_hit_rate=True, pinned_buf_size=1_000_000)

    dl_processes = []
    for i in range(num_engines):
        dl_processes.append(PipelinedDataloader(feature_stores[i], torch.device('cuda', i), num_nodes))
        
    [p.start() for p in dl_processes]
    [p.join() for p in dl_processes]