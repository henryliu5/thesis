from torch.multiprocessing import Process, Queue, Condition, Barrier
from multiprocessing import shared_memory
from enum import Enum
from fast_inference.request_generator import Request, RequestType
from fast_inference.sampler import InferenceSampler
from fast_inference.feat_server import FeatureServer, ManagedCacheServer
from fast_inference.util import create_feature_stores
from typing import Dict
import torch
import dgl
from dgl.utils import get_numa_nodes_cores
import psutil
import os
import time

class WorkerType(Enum):
    DISABLED = 0
    SAMPLER = 1
    DATA_LOADER = 2
    MODEL_EXECUTOR = 3

    def __str__(self):
        return str(self.name)

def serialize_mfgs(mfgs):
    # Need to deconstruct and COPY mfg nids
    us = []
    vs = []
    num_src_nodes = []
    num_dst_nodes = []
    for mfg in mfgs:
        u,v = mfg.edges()
        us.append(u.clone())
        vs.append(v.clone())
        num_src_nodes.append(mfg.num_src_nodes())
        num_dst_nodes.append(mfg.num_dst_nodes())

    return (us, vs, num_src_nodes, num_dst_nodes)

def deserialize_mfgs(mfg_tuple):
    us, vs, num_src_nodes, num_dst_nodes = mfg_tuple
    mfgs = []
    for i in range(len(us)):
        mfgs.append(dgl.create_block((us[i], vs[i]), num_src_nodes=num_src_nodes[i], num_dst_nodes=num_dst_nodes[i]))
    return mfgs


class PipelineManager(Process):
    """
    Can control state of pipeline workers by interacting with numpy array in shared memory
    """

    def __init__(self, num_workers: int, response_queue: Queue):
        super().__init__()
        self.response_queue = response_queue
        
        # self.worker_types = shared_memory.SharedMemory(create=True, size=a)

    def run(self):
        pass


class PipelineWorker(Process):
    def __init__(self, sampler: InferenceSampler,
                 feature_store: FeatureServer,
                 model: torch.nn.Module,
                 request_queue: Queue,
                 sampled_mfgs_queue: Queue,
                 feature_queue: Queue,
                 response_queue: Queue,
                 disable_cv: Condition,
                 barriers: Dict[str, Barrier],
                 device: torch.device,
                 worker_type: WorkerType):
        super().__init__()

        self.sampler = sampler
        self.feature_store = feature_store
        self.model = model

        self.request_queue = request_queue
        self.sampled_mfgs_queue = sampled_mfgs_queue
        self.feature_queue = feature_queue
        self.response_queue = response_queue
        self.disable_cv = disable_cv

        self.barriers = barriers
        self.device = device
        self.worker_type = worker_type

        print('Creating worker', self.worker_type, 'on device', self.device)

    @torch.inference_mode()
    def run(self):
        numa_info = get_numa_nodes_cores()
        pin_cores = [cpus[0] for core_id, cpus in numa_info[self.device.index % 2]]

        psutil.Process().cpu_affinity(pin_cores)
        print(f'setting cpu affinity', psutil.Process().cpu_affinity())
        # torch.set_num_threads(os.cpu_count() // 2)
        torch.set_num_threads(4)
        print('using intra-op threads:', torch.get_num_threads())

        if type(self.feature_store) == ManagedCacheServer:
            self.feature_store.start_manager()

        update_window = 5
        requests_handled = 2

        self.barriers['start'].wait()
        while True:
            if self.worker_type == WorkerType.DISABLED:
                self.disable_cv.wait()
                # Upon wakeup, should not be disabled anymore
                # TODO add check here by reading shm
            elif self.worker_type == WorkerType.SAMPLER:
                req = self.request_queue.get()
                s = time.perf_counter()

                mfgs = self.sampler.sample(
                    req.nids, req.edges, use_gpu_sampling=True, device=self.device)
                
                required_feats = mfgs[0].ndata['_ID']['_N']
                self.sampled_mfgs_queue.put((required_feats, serialize_mfgs(mfgs)))

                e = time.perf_counter()
                # print('sample', e-s)

            elif self.worker_type == WorkerType.DATA_LOADER:
                if requests_handled % update_window == 0:
                    self.feature_store.compute_topk()
                    self.feature_store.update_cache(['feat'])

                required_feats, mfg_tuple = self.sampled_mfgs_queue.get()
                s = time.perf_counter()

                mfgs = deserialize_mfgs(mfg_tuple)
                inputs, mfgs = self.feature_store.get_features(
                    required_feats, feats=['feat'], mfgs=mfgs)
                inputs = inputs['feat']

                self.feature_queue.put((inputs, serialize_mfgs(mfgs)))
                requests_handled += 1
                e = time.perf_counter()
                # print('gather', e-s)


            elif self.worker_type == WorkerType.MODEL_EXECUTOR:
                inputs, mfg_tuple = self.feature_queue.get()
                s = time.perf_counter()

                mfgs = deserialize_mfgs(mfg_tuple)
                outputs = self.model(mfgs, inputs)
                outputs = outputs.cpu()

                # self.response_queue.put(Request(None, None, None, req.id, req.trial, RequestType.RESPONSE if req.req_type == RequestType.INFERENCE else req.req_type, req.time_generated, req.time_exec_started, req.time_exec_finished))
                self.response_queue.put(outputs)

                e = time.perf_counter()
                # print('exec', e-s)


def create_pipeline(num_devices: int, samplers: int, data_loaders: int, model_executors: int,
                    cache_type: str, 
                    cache_percent: float,
                    g: dgl.DGLGraph,
                    model: torch.nn.Module,
                    request_queue: Queue,
                    response_queue: Queue,
                    barriers: Dict[str, Barrier]):
    """
    Creates a pipeline of processes that will run inference in parallel
    """
    workers_per_device = samplers + data_loaders + model_executors

    feature_stores = create_feature_stores(cache_type=cache_type,
                                           num_stores=num_devices,
                                           executors_per_store=data_loaders,
                                           graph=g,
                                           track_feature_types=['feat'],
                                           cache_percent=cache_percent,
                                           use_pinned_mem=True,
                                           profile_hit_rate=True,
                                           pinned_buf_size=1_000_000)

    logical_g = dgl.graph(g.edges())

    disable_cv = None
    workers = []

    for device_id in range(num_devices):
        logical_g = logical_g.to(torch.device('cuda', device_id))
        sampled_mfgs_queue = Queue(4)
        feature_queue = Queue(4)
        model = model.to(torch.device('cuda', device_id))

        for i in range(workers_per_device):
            worker_type = WorkerType.SAMPLER
            feature_store = None
            if i >= samplers + data_loaders:
                worker_type = WorkerType.MODEL_EXECUTOR
                feature_store = None
            elif i >= samplers:
                worker_type = WorkerType.DATA_LOADER
                feature_store = feature_stores[device_id][i - samplers]

            sampler = InferenceSampler(logical_g)
            worker = PipelineWorker(sampler, feature_store, model, request_queue,
                                    sampled_mfgs_queue, feature_queue, response_queue, disable_cv, barriers, 
                                    torch.device('cuda', device_id), worker_type)
            
            worker.start()
            workers.append(worker)
    return workers