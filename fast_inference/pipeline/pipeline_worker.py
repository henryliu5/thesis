import torch
from fast_inference.request_generator import Request, RequestType
import time
from dataclasses import dataclass
from typing import Tuple, Dict

from torch.multiprocessing import Process, Queue, Condition, Barrier
from multiprocessing import shared_memory
from enum import Enum
from fast_inference.sampler import InferenceSampler
from fast_inference.feat_server import FeatureServer, ManagedCacheServer
from typing import Dict
import dgl
from dgl.utils import get_numa_nodes_cores
import psutil
import time
import numpy as np


def serialize_mfgs(mfgs):
    # Need to deconstruct and COPY mfg nids
    us = []
    vs = []
    num_src_nodes = []
    num_dst_nodes = []
    for mfg in mfgs:
        u, v = mfg.edges()
        us.append(u.clone())
        vs.append(v.clone())
        num_src_nodes.append(mfg.num_src_nodes())
        num_dst_nodes.append(mfg.num_dst_nodes())

    return (us, vs, num_src_nodes, num_dst_nodes)


def deserialize_mfgs(mfg_tuple):
    us, vs, num_src_nodes, num_dst_nodes = mfg_tuple
    mfgs = []
    for i in range(len(us)):
        mfgs.append(dgl.create_block(
            (us[i], vs[i]), num_src_nodes=num_src_nodes[i], num_dst_nodes=num_dst_nodes[i]))
    return mfgs


@dataclass(frozen=True)
class SamplerQueueMessage:
    id: int
    required_feats: torch.Tensor
    mfg_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    timing_info: Dict[str, float]


@dataclass(frozen=True)
class FeatureQueueMessage:
    id: int
    inputs: torch.Tensor
    mfg_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    timing_info: Dict[str, float]


class ControlMessage(Enum):
    SHUTDOWN = 0

    def __str__(self):
        return str(self.name)


class WorkerType(Enum):
    DISABLED = 0
    SAMPLER = 1
    DATA_LOADER = 2
    MODEL_EXECUTOR = 3
    SHUTDOWN = 4

    def __str__(self):
        return str(self.name)


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
                 worker_type_buf: shared_memory.SharedMemory,
                 worker_id: int,
                 num_workers: int,
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

        self.worker_type_arr = np.ndarray(
            (num_workers,), np.int32, worker_type_buf.buf)
        # self.worker_type_arr = torch.frombuffer(worker_type_buf.buf, dtype=torch.int32, count=num_workers)
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.worker_type = worker_type

        self.update_window = 5
        self.requests_handled = 2
        print('Creating worker', self.worker_type, 'on device', self.device)

    @torch.inference_mode()
    def run(self):
        if self.feature_store is not None:
            for k, v in self.feature_store.pinned_buf_dict.items():
                self.feature_store.pinned_buf_dict[k] = v.pin_memory()

        numa_info = get_numa_nodes_cores()
        pin_cores = [cpus[0]
                     for core_id, cpus in numa_info[self.device.index % 2]]

        psutil.Process().cpu_affinity(pin_cores)
        print(f'setting cpu affinity', psutil.Process().cpu_affinity())
        # torch.set_num_threads(os.cpu_count() // 2)
        torch.set_num_threads(16)
        print('using intra-op threads:', torch.get_num_threads())

        if type(self.feature_store) == ManagedCacheServer:
            self.feature_store.start_manager()

        self.barriers['start'].wait()
        while True:
            self.worker_type = WorkerType(self.worker_type_arr[self.worker_id])
            if self.worker_type == WorkerType.DISABLED:
                self.disable_cv.wait()
                # Upon wakeup, should not be disabled anymore
                # TODO add check here by reading shm
            elif self.worker_type == WorkerType.SAMPLER:
                req = self.request_queue.get()
                if type(req) == ControlMessage:
                    self.sampled_mfgs_queue.put(req)
                    break

                self.sampled_mfgs_queue.put(sampler_strategy(self, req))

            elif self.worker_type == WorkerType.DATA_LOADER:
                in_msg = self.sampled_mfgs_queue.get()
                if type(in_msg) == ControlMessage:
                    self.feature_queue.put(in_msg)
                    break

                self.feature_queue.put(data_loader_strategy(self, in_msg))

            elif self.worker_type == WorkerType.MODEL_EXECUTOR:
                in_msg = self.feature_queue.get()
                if type(in_msg) == ControlMessage:
                    self.response_queue.put(in_msg)
                    break

                self.response_queue.put(model_executor_strategy(self, in_msg))

            elif self.worker_type == WorkerType.SHUTDOWN:
                break

        self.cleanup()
        print('Worker', self.worker_id, 'exiting')

    def cleanup(self):
        print('Worker', self.worker_id, 'cleaning up CUDA tensors')
        del self.model
        del self.sampler
        del self.feature_store
        # self.barriers['finish'].wait()


def sampler_strategy(worker: PipelineWorker, in_msg: Request) -> FeatureQueueMessage:
    req = in_msg

    s = time.perf_counter()
    mfgs = worker.sampler.sample(
        req.nids, req.edges, use_gpu_sampling=True, device=worker.device)

    required_feats = mfgs[0].ndata['_ID']['_N']
    e = time.perf_counter()

    return SamplerQueueMessage(req.id, required_feats, serialize_mfgs(mfgs), {'sample': e - s, 'time_generated': req.time_generated, 'sample_start': s})


def data_loader_strategy(worker: PipelineWorker, in_msg: SamplerQueueMessage) -> FeatureQueueMessage:
    s = time.perf_counter()

    if worker.requests_handled % worker.update_window == 0:
        worker.feature_store.update_cache()

    mfgs = deserialize_mfgs(in_msg.mfg_tuple)
    inputs, mfgs = worker.feature_store.get_features(
        in_msg.required_feats, feats=['feat'], mfgs=mfgs)
    inputs = inputs['feat']

    e = time.perf_counter()
    worker.requests_handled += 1
    return FeatureQueueMessage(in_msg.id, inputs, serialize_mfgs(mfgs), {'data_load_time': e - s} | in_msg.timing_info)


def model_executor_strategy(worker: PipelineWorker, in_msg: FeatureQueueMessage) -> Request:
    msg = in_msg
    s = time.perf_counter()

    mfgs = deserialize_mfgs(msg.mfg_tuple)
    outputs = worker.model(mfgs, msg.inputs)
    outputs = outputs.cpu()

    e = time.perf_counter()
    print('elapsed', round(e - msg.timing_info['sample_start'], 5),
          round(e - msg.timing_info['time_generated'], 5),
          'sample', round(msg.timing_info['sample'], 5),
          'data_load', round(msg.timing_info['data_load_time'], 5),
          'exec', round(e - s, 5),
          'diff', round(e - msg.timing_info['sample_start'] - msg.timing_info['sample'] - msg.timing_info['data_load_time'] - (e-s), 5))
    return Request(None, None, None, msg.id, None, RequestType.RESPONSE, msg.timing_info['time_generated'], None, None)
