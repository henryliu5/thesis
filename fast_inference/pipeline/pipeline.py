from torch.multiprocessing import Queue, Barrier
from multiprocessing import shared_memory
from fast_inference.sampler import InferenceSampler
from fast_inference.util import create_feature_stores
from typing import Dict
import torch
import dgl
from fast_inference.pipeline.pipeline_worker import PipelineWorker, WorkerType
import numpy as np


class PipelineManager:
    """
    Can control state of pipeline workers by interacting with numpy array in shared memory
    """

    def __init__(self, samplers: int, data_loaders: int, executors: int, num_devices: int, response_queue: Queue):

        workers_per_device = samplers + data_loaders + executors
        num_workers = workers_per_device * num_devices

        # a = torch.zeros(num_workers, dtype=torch.int32)
        a = np.zeros(num_workers, dtype=np.int32)

        for device_id in range(num_devices):
            for i in range(workers_per_device):
                worker_id = device_id * workers_per_device + i

                worker_type = WorkerType.SAMPLER
                if i >= samplers + data_loaders:
                    worker_type = WorkerType.MODEL_EXECUTOR
                elif i >= samplers:
                    worker_type = WorkerType.DATA_LOADER

                a[worker_id] = worker_type.value

        self.worker_types_buf = shared_memory.SharedMemory(
            create=True, size=a.nbytes)
        # self.worker_types_buf = shared_memory.SharedMemory(create=True, size=a.numel() * a.element_size())
        self.worker_types_np = np.ndarray(
            a.shape, dtype=a.dtype, buffer=self.worker_types_buf.buf)
        # self.worker_types_np = torch.frombuffer(self.worker_types_buf.buf, dtype=torch.int32, count=a.numel()).reshape(a.shape)
        self.worker_types_np[:] = a[:]

    def shutdown(self):
        self.worker_types_np[:] = WorkerType.SHUTDOWN.value


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
    if torch.multiprocessing.get_start_method() != 'spawn':
        print('Start method must be set to start to build pipeline, exiting')
        exit()

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

    total_workers = num_devices * workers_per_device

    manager = PipelineManager(samplers, data_loaders,
                              model_executors, num_devices, response_queue)

    workers = []

    for device_id in range(num_devices):
        logical_g = logical_g.to(torch.device('cuda', device_id))
        sampled_mfgs_queue = Queue(1)
        feature_queue = Queue(1)
        model = model.to(torch.device('cuda', device_id))

        for i in range(workers_per_device):
            worker_id = device_id * workers_per_device + i

            worker_type = WorkerType(manager.worker_types_np[worker_id])
            if worker_type == WorkerType.DATA_LOADER:
                feature_store = feature_stores[device_id][i - samplers]
            else:
                feature_store = None

            sampler = InferenceSampler(logical_g)
            worker = PipelineWorker(sampler, feature_store, model, request_queue,
                                    sampled_mfgs_queue, feature_queue, response_queue, disable_cv, barriers,
                                    torch.device('cuda', device_id),
                                    worker_type_buf=manager.worker_types_buf,
                                    worker_id=worker_id,
                                    num_workers=total_workers,
                                    worker_type=worker_type)

            worker.start()
            workers.append(worker)

    return manager, workers
