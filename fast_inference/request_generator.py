import torch
from torch.multiprocessing import Process, Queue, Barrier
from fast_inference.dataset import InferenceTrace
from fast_inference.timer import Timer, enable_timers, clear_timers, print_timer_info, export_timer_info, export_dict_as_pd
from tqdm import tqdm
import time
from fast_inference.dataset import InferenceDataset, FastEdgeRepr
from fast_inference.message import Message, MessageType, RequestPayload
import numpy as np


class RequestGenerator(Process):
    def __init__(self, request_queue: Queue, start_barrier: Barrier, finish_barrier: Barrier, trial_barriers: Barrier, num_engines: int, trace: InferenceTrace, batch_size: int, max_iters: int, rate: float, trials: int):
        super().__init__()
        self.request_queue = request_queue
        self.start_barrier = start_barrier
        self.finish_barrier = finish_barrier
        self.trial_barriers = trial_barriers
        self.num_engines = num_engines

        self.trace = trace
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.rate = rate
        self.trials = trials

    @torch.inference_mode()
    def run(self):
        # infer_data = InferenceDataset('ogbn-products', 0.1, partitions=5, force_reload=False, verbose=True)
        # self.trace = infer_data.create_inference_trace(subgraph_bias=None)

        torch.set_num_threads(1)
        # if self.rate == 0:
        #     delay_between_requests = 0
        # else:
        #     delay_between_requests = 1 / self.rate
        enable_timers()
        
        print('Request generator waiting')
        self.start_barrier.wait()
        n = len(self.trace)

        # Warmup batches
        WARMUPS = 3
        print('Doing warmups...')
        for _ in range(WARMUPS):
            for i in tqdm(range(0, min(n, 150 * self.batch_size), self.batch_size)):
                with Timer('send batch'):
                    if i + self.batch_size >= n:
                        continue
                    nids = self.trace.nids[i:i+self.batch_size]
                    features = self.trace.features[i:i+self.batch_size]
                    edges = self.trace.edges.get_batch(i, i + self.batch_size)

                    # req = Request(nids, features, edges, i, None, RequestType.WARMUP, time.perf_counter())
                    request = Message(id=i, 
                                      trial=None,
                                      timing_info={'time_generated': time.perf_counter()},
                                      msg_type=MessageType.WARMUP,
                                      payload=RequestPayload(nids=nids,
                                                             features=features,
                                                             edges=edges)
                                      )
                    self.request_queue.put(request)

        while self.request_queue.qsize() > 0:
            time.sleep(1)
        
        for trial in range(self.trials):
            print('Starting trial', trial)
            clear_timers()
            for i in tqdm(range(0, min(n, self.max_iters * self.batch_size), self.batch_size)):
                with Timer('send batch'):
                    if i + self.batch_size >= n:
                        continue
                    batch_start = time.perf_counter()
                    nids = self.trace.nids[i:i+self.batch_size]
                    features = self.trace.features[i:i+self.batch_size]
                    edges = self.trace.edges.get_batch(i, i + self.batch_size)
                    batch_end = time.perf_counter()

                    if self.rate != 0:
                        delay = np.random.exponential(1 / self.rate)
                        time.sleep(max(delay - (batch_end - batch_start), 0))

                    # req = Request(nids, features, edges, i, trial, RequestType.INFERENCE, time.perf_counter())
                    request = Message(id=i,
                                      trial=trial,
                                      timing_info={'time_generated': time.perf_counter()},
                                      msg_type=MessageType.INFERENCE,
                                      payload=RequestPayload(nids=nids,
                                                             features=features,
                                                             edges=edges)
                                      )
                    self.request_queue.put(request)

            # Send out resets
            for i in range(self.num_engines):
                print('sending reset', i)
                # self.request_queue.put(Request(None, None, None, None, None, RequestType.RESET, None))
                self.request_queue.put(Message(id=-1,
                                                trial=-1,
                                                timing_info={},
                                                msg_type=MessageType.RESET))

            self.trial_barriers[trial].wait(60)
            if self.trial_barriers[trial].broken:
                print('Barrier broken due to timeout')
                exit()
            time.sleep(1)

        # Need to have different shutdown mechanism
        for i in range(self.num_engines):
            # self.request_queue.put(Request(None, None, None, None, None, RequestType.SHUTDOWN, None))
            self.request_queue.put(Message(id=-1,
                                           trial=-1,
                                           timing_info={},
                                           msg_type=MessageType.SHUTDOWN))

        self.finish_barrier.wait()

from fast_inference.timer import TRACES

class ResponseRecipient(Process):
    def __init__(self, response_queue: Queue, start_barrier: Barrier, finish_barrier: Barrier, trial_barriers: Barrier, num_engines: int, num_devices:int, executors_per_store, dataset, model_name, batch_size, output_path):
        super().__init__()
        self.response_queue = response_queue
        self.start_barrier = start_barrier
        self.finish_barrier = finish_barrier
        self.trial_barriers = trial_barriers
        self.num_engines = num_engines

        # Benchmarking info
        self.dataset = dataset
        self.model_name = model_name
        self.batch_size = batch_size
        self.output_path = output_path
        self.num_devices = num_devices
        self.executors_per_store = executors_per_store
    
    def run(self):
        torch.set_num_threads(1)
        global TRACES
        TRACES['total'] = []
        TRACES['total (received)'] = []
        TRACES['id'] = []

        print('Response recipient waiting')
        self.start_barrier.wait()
        cur_trial = 0
        num_resets = 0
        while True:
            msg: Message = self.response_queue.get()
            if msg.msg_type == MessageType.RESPONSE:
                time_received = time.perf_counter()
                TRACES['total (received)'].append(time_received - msg.timing_info['time_generated'])
                TRACES['total'].append(msg.timing_info['time_exec_finished'] - msg.timing_info['time_exec_started'])
                TRACES['id'].append(msg.id)

                # print('got resp for', msg.id, 'in', time.perf_counter() - msg.time_generated)
                assert(msg.trial == cur_trial)
            elif msg.msg_type == MessageType.SHUTDOWN:
                print('ResponseRecipient received shutdown')
                break
            elif msg.msg_type == MessageType.RESET:
                # Finished 1 trial, need to receive finish from all engines
                num_resets += 1
                if num_resets == self.num_engines:
                    print_timer_info()
                    export_timer_info(f'{self.output_path}/{self.model_name.upper()}', {'name': self.dataset, 'batch_size': self.batch_size, 'trial': cur_trial})
                    self.trial_barriers[cur_trial].wait()
                    cur_trial += 1
                    num_resets = 0

                    # TODO make this use id to determine number of request handled
                    throughput = self.num_engines * (1 / (sum(TRACES['total']) / len(TRACES['total'])))
                    print('reqsuests handled', len(TRACES['total']))
                    print('avg handle time', (sum(TRACES['total']) / len(TRACES['total'])))
                    print('--------------throughput', throughput)
                    export_dict_as_pd({'throughput (req/s)': [throughput]}, f'{self.output_path}/{self.model_name.upper()}_throughput', {'name': self.dataset, 'batch_size': self.batch_size, 'trial': cur_trial, 'num_devices': self.num_devices, 'executors_per_store': self.executors_per_store}, ignore_first_n=0)

                    clear_timers()
        
        self.finish_barrier.wait()


        