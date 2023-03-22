import torch
from torch.multiprocessing import Process, Queue, Barrier
from fast_inference.dataset import InferenceTrace
from typing import List, Dict
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm
import time
from fast_inference.dataset import InferenceDataset, FastEdgeRepr
 
class RequestType(Enum):
    INFERENCE = 1
    RESET = 2
    SHUTDOWN = 3
    RESPONSE = 4

@dataclass(frozen=True)
class Request:
    nids: torch.Tensor
    features: torch.Tensor
    edges: FastEdgeRepr

    id: int
    trial: int
    req_type: RequestType
    start_time: float


class RequestGenerator(Process):
    def __init__(self, request_queue: Queue, start_barrier: Barrier, finish_barrier: Barrier, trace: InferenceTrace, batch_size: int, max_iters: int, rate: float, trials: int):
        super().__init__()
        self.request_queue = request_queue
        self.start_barrier = start_barrier
        self.finish_barrier = finish_barrier

        self.trace = trace
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.rate = rate
        self.trials = trials

    def run(self):
        # infer_data = InferenceDataset('ogbn-products', 0.1, partitions=5, force_reload=False, verbose=True)
        # self.trace = infer_data.create_inference_trace(subgraph_bias=None)

        torch.set_num_threads(1)
        if self.rate == 0:
            delay_between_requests = 0
        else:
            delay_between_requests = 1 / self.rate
        print('Request generator waiting')
        self.start_barrier.wait()

        n = len(self.trace)
        for trial in range(self.trials):
            print('Starting trial', trial)
            for i in tqdm(range(0, min(n, self.max_iters * self.batch_size), self.batch_size)):
                if i + self.batch_size >= n:
                    continue
                batch_start = time.perf_counter()
                nids = self.trace.nids[i:i+self.batch_size]
                features = self.trace.features[i:i+self.batch_size]
                edges = self.trace.edges.get_batch(i, i + self.batch_size)
                batch_end = time.perf_counter()
                time.sleep(max(delay_between_requests - (batch_end - batch_start), 0))

                req = Request(nids, features, edges, i, trial, RequestType.INFERENCE, time.perf_counter())
                self.request_queue.put(req)

            self.request_queue.put(Request(None, None, None, None, None, RequestType.RESET, None))

        self.request_queue.put(Request(None, None, None, None, None, RequestType.SHUTDOWN, None))

        self.finish_barrier.wait()

from fast_inference.timer import TRACES

class ResponseRecipient(Process):
    def __init__(self, response_queue: Queue, start_barrier: Barrier):
        super().__init__()
        self.response_queue = response_queue
        self.start_barrier = start_barrier
    
    def run(self):
        torch.set_num_threads(1)
        global TRACES
        TRACES['total'] = []
        TRACES['id'] = []

        print('Response recipient waiting')
        self.start_barrier.wait()
        while True:
            resp = self.response_queue.get()
            if resp.req_type == RequestType.RESPONSE:
                TRACES['total'].append(time.perf_counter() - resp.start_time)
                TRACES['id'].append(resp.id)

                print('got resp for', resp.id, 'in', time.perf_counter() - resp.start_time)
            elif resp.req_type == RequestType.SHUTDOWN:
                exit('ResponseRecipient received shutdown')


        