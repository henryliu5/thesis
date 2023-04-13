import torch
from fast_inference.message import Message, MessageType, RequestPayload, SamplerQueuePayload, FeatureQueuePayload, ResponsePayload
from fast_inference.sampler import InferenceSampler
from fast_inference.feat_server import FeatureServer, ManagedCacheServer
from fast_inference.timer import export_dict_as_pd, enable_timers, Timer
import time
from dataclasses import dataclass
from typing import Tuple, Dict, Callable

from torch.multiprocessing import Process, Queue, Condition, Barrier
from multiprocessing import shared_memory
from enum import Enum

import dgl
from dgl.utils import get_numa_nodes_cores
import psutil
import time
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
from contextlib import nullcontext

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

        enable_timers()
    
        use_prof = False
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) if use_prof else nullcontext() as prof:
            self.barriers['start'].wait()
            while True:
                self.worker_type = WorkerType(self.worker_type_arr[self.worker_id])
                if self.worker_type == WorkerType.DISABLED:
                    self.disable_cv.wait()
                    # Upon wakeup, should not be disabled anymore
                    # TODO add check here by reading shm
                elif self.worker_type == WorkerType.SAMPLER:
                    in_queue = self.request_queue
                    out_queue = self.sampled_mfgs_queue
                    strategy = sampler_strategy

                elif self.worker_type == WorkerType.DATA_LOADER:
                    in_queue = self.sampled_mfgs_queue
                    out_queue = self.feature_queue
                    strategy = data_loader_strategy

                elif self.worker_type == WorkerType.MODEL_EXECUTOR:
                    in_queue = self.feature_queue
                    out_queue = self.response_queue
                    strategy = model_executor_strategy

                elif self.worker_type == WorkerType.SHUTDOWN:
                    break

                should_exit = self.dispatch(in_queue, out_queue, strategy)
                if should_exit:
                    break

        if use_prof:
            prof.export_chrome_trace(f'pipeline_trace_rank_{self.worker_id}_{self.worker_type.name}_total_{self.num_workers}.json')
        self.cleanup()
        print('Worker', self.worker_id, 'exiting')

    def cleanup(self):
        if self.feature_store is not None:
            export_dict_as_pd(self.feature_store.lock_conflict_trace, 'pipeline_trace', {'cache_type': self.feature_store.cache_name,
                                                                                     'store_id': self.feature_store.store_id,
                                                                                     'executor_id': self.feature_store.executor_id}, 0)

        print('Worker', self.worker_id, 'cleaning up CUDA tensors')
        del self.model
        del self.sampler
        del self.feature_store
        # self.barriers['finish'].wait()

    def dispatch(self, in_queue: Queue, out_queue: Queue, strategy: Callable) -> bool:
        msg: Message = in_queue.get()
        should_exit = (msg.msg_type == MessageType.SHUTDOWN) or (msg.msg_type == MessageType.RESET)
        if not should_exit:
            with Timer('strategy', track_cuda=True):
                out_msg = strategy(self, msg)
            del msg
            out_queue.put(out_msg)
        else:
            out_queue.put(msg)
        return should_exit
    
        # try:
        #     msg: Message = in_queue.get(timeout=5)
        #     out_queue.put(strategy(self, msg))
        # except Empty as error:
        #     pass

def sampler_strategy(worker: PipelineWorker, in_msg: Message) -> Message:
    payload: RequestPayload = in_msg.payload
    assert (isinstance(payload, RequestPayload)), f'Expected RequestPayload, got {type(payload)}, message type {in_msg.msg_type}'

    s = time.perf_counter()
    mfgs = worker.sampler.sample(
        payload.nids, payload.edges, use_gpu_sampling=True, device=worker.device)

    required_feats = mfgs[0].ndata['_ID']['_N']
    e = time.perf_counter()

    return Message(id=in_msg.id,
                   trial=in_msg.trial,
                   timing_info=in_msg.timing_info | {
                       'sample': e - s, 'sample_start': s},
                   msg_type=in_msg.msg_type,
                   payload=SamplerQueuePayload(required_feats, serialize_mfgs(mfgs)))


def data_loader_strategy(worker: PipelineWorker, in_msg: Message) -> Message:
    payload: SamplerQueuePayload = in_msg.payload
    assert (isinstance(payload, SamplerQueuePayload)), f'Expected SamplerQueuePayload, got {type(payload)}, message type {in_msg.msg_type}'

    s = time.perf_counter()
    if worker.requests_handled % worker.update_window == 0:
        worker.feature_store.update_cache()

    with Timer('deserialize mfgs'):
        mfgs = deserialize_mfgs(payload.mfg_tuple)
    inputs, mfgs = worker.feature_store.get_features(
        payload.required_feats, feats=['feat'], mfgs=mfgs)
    inputs = inputs['feat']

    e = time.perf_counter()
    worker.requests_handled += 1

    return Message(id=in_msg.id,
                   trial=in_msg.trial,
                   timing_info=in_msg.timing_info | {'data_load_time': e - s},
                   msg_type=in_msg.msg_type,
                   payload=FeatureQueuePayload(inputs, serialize_mfgs(mfgs)))


def model_executor_strategy(worker: PipelineWorker, in_msg: Message) -> Message:
    msg = in_msg
    payload: FeatureQueuePayload = msg.payload
    assert(isinstance(payload, FeatureQueuePayload)), f'Expected FeatureQueuePayload, got {type(payload)}, message type {msg.msg_type}'

    s = time.perf_counter()

    mfgs = deserialize_mfgs(payload.mfg_tuple)
    outputs = worker.model(mfgs, payload.inputs)
    outputs = outputs.cpu()

    e = time.perf_counter()
    print('elapsed', round(e - msg.timing_info['sample_start'], 5),
          round(e - msg.timing_info['time_generated'], 5),
          'sample', round(msg.timing_info['sample'], 5),
          'data_load', round(msg.timing_info['data_load_time'], 5),
          'exec', round(e - s, 5),
          'diff', round(e - msg.timing_info['sample_start'] - msg.timing_info['sample'] - msg.timing_info['data_load_time'] - (e-s), 5))

    return Message(id=msg.id,
                   trial=msg.trial,
                   timing_info=msg.timing_info | {'exec_time': e - s},
                   msg_type=MessageType.RESPONSE,
                   payload=ResponsePayload())
