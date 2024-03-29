import torch
from torch.multiprocessing import Process, Queue, Barrier
from fast_inference.message import Message, MessageType, RequestPayload
from fast_inference.request_generator import MessageType
from fast_inference.feat_server import FeatureServer, ManagedCacheServer, CountingFeatServer
from fast_inference.sampler import InferenceSampler
from fast_inference.timer import Timer, enable_timers, clear_timers, print_timer_info, export_dict_as_pd, export_timer_info, TRACES
from torch.profiler import profile, record_function, ProfilerActivity
from contextlib import nullcontext
import time
from copy import deepcopy
import os
import psutil
from dgl.utils.internal import get_numa_nodes_cores

class InferenceEngine(Process):
    def __init__(self, request_queue: Queue,
                 response_queue: Queue,
                 start_barrier: Barrier,
                 finish_barrier: Barrier,
                 trial_barriers: Barrier,
                 num_engines: int,
                 device: torch.device,
                 feature_store: FeatureServer,
                 logical_g,
                 model,
                 dataset, model_name, batch_size, output_path):
        
        super().__init__()
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.start_barrier = start_barrier
        self.finish_barrier = finish_barrier
        self.trial_barriers = trial_barriers
        self.num_engines = num_engines
        self.device = device
        self.device_id = device.index
        self.feature_store = feature_store

        self.logical_g = logical_g
        self.sampler = InferenceSampler(logical_g)
        self.model = model.to(device)

        # Benchmarking info
        self.dataset = dataset
        self.model_name = model_name
        self.batch_size = batch_size
        self.output_path = output_path
    
    @torch.inference_mode()
    def run(self):
        numa_info = get_numa_nodes_cores()
        # Pin just to first cpu in each core in numa node 0 (DGL approach)
        pin_cores = [cpus[0] for core_id, cpus in numa_info[self.device_id % 2]]
        # pin_cores = [cpu for core_id, cpus in numa_info[self.device_id % 2] for cpu in cpus]
        psutil.Process().cpu_affinity(pin_cores)
        print(f'engine {self.device_id}, cpu affinity', psutil.Process().cpu_affinity())

        assert(torch.is_inference_mode_enabled())
        # TODO change to num cpu threads / num inference engine
        # print(os.cpu_count()) # 64 on mew
        # torch.set_num_threads(os.cpu_count() // self.num_engines // 2)
        torch.set_num_threads(os.cpu_count() // self.feature_store.executors_per_store // 2)
        # # 3/25 I don't think changing interop should make difference
        # torch.set_num_interop_threads(os.cpu_count() // self.feature_store.executors_per_store)
        # torch.set_num_interop_threads()
        print('using intra-op threads:', torch.get_num_threads())
        # print('using inter-op threads', torch.get_num_interop_threads())

        # Need to re-pin the feature store buffer
        for k, v in self.feature_store.pinned_buf_dict.items():
            self.feature_store.pinned_buf_dict[k] = v.pin_memory()

        if type(self.feature_store) == ManagedCacheServer:
            self.feature_store.start_manager()
        elif type(self.feature_store) == CountingFeatServer:
            self.feature_store.init_locks()

        TRACES['exec_time_since_generated'] = []
        
        self.model_stream = torch.cuda.Stream(device=self.device, priority=-1)
        print('InferenceEngine', self.device_id, 'started')
        self.start_barrier.wait()

        update_window = 5# * self.num_engines
        requests_handled = 2
    
        use_prof = False
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) if use_prof else nullcontext() as prof:
            enable_timers()
            cur_trial = 0
            with torch.cuda.device(self.device): # needed to set timers on correct device
                while True:
                    req = self.request_queue.get()
                    if req.msg_type == MessageType.RESET:
                        print('engine', self.device_id, 'received reset')

                    timing_info = req.timing_info

                    timing_info['time_exec_started'] = time.perf_counter()
                    if req.msg_type == MessageType.INFERENCE or req.msg_type == MessageType.WARMUP:
                        with Timer('exec request'):
                            if requests_handled % update_window == 0:
                                self.feature_store.update_cache()

                            assert(isinstance(req.payload, RequestPayload))
                            payload = req.payload
                            mfgs = self.sampler.sample(payload.nids, payload.edges, use_gpu_sampling=True, device=self.device)

                            with Timer('dataloading', track_cuda=True):
                                required_feats = mfgs[0].ndata['_ID']['_N']
                                inputs, mfgs = self.feature_store.get_features(required_feats, feats=['feat'], mfgs=mfgs)
                                inputs = inputs['feat']
                            
                            with Timer('model'):
                                with torch.cuda.stream(self.model_stream):
                                    x = self.model(mfgs, inputs)
                                    x.cpu()
                                torch.cuda.current_stream().wait_stream(self.model_stream)
                        requests_handled += 1
                        TRACES['exec_time_since_generated'].append(time.perf_counter() - timing_info['time_generated'])
                        
                    timing_info['time_exec_finished'] = time.perf_counter()
                   

                    if req.msg_type != MessageType.WARMUP:
                        # self.response_queue.put(Request(None, None, None, req.id, req.trial, MessageType.RESPONSE if req.msg_type == MessageType.INFERENCE else req.msg_type, req.time_generated, req.time_exec_started, req.time_exec_finished))
                        self.response_queue.put(Message(id=req.id,
                                                         trial=req.trial, 
                                                         timing_info=timing_info, 
                                                         msg_type=MessageType.RESPONSE if req.msg_type == MessageType.INFERENCE else req.msg_type))
                    if req.msg_type == MessageType.SHUTDOWN:
                        print(f'InferenceEngine {self.device_id} received shutdown request, id: {req.id}')
                        break
                    if req.msg_type == MessageType.RESET:
                        if self.output_path != None and self.device_id == 0:
                            self.feature_store.export_profile(f'{self.output_path}/{self.model_name.upper()}_cache_info', {'name': self.dataset, 'batch_size': self.batch_size, 'trial': cur_trial})
                        # print_timer_info()    
                        #!! TODO replace GCN with model name
                        export_timer_info(f'{self.output_path}/GCN_breakdown_with_trials', {'cache_type': self.feature_store.cache_name,
                                                                            'store_id': self.feature_store.store_id,
                                                                            'executor_id': self.feature_store.executor_id,
                                                                            'num_stores': len(self.feature_store.caches),
                                                                            'executors_per_store': self.feature_store.executors_per_store,
                                                                            'trial': cur_trial})

                        export_timer_info(f'{self.output_path}/GCN_breakdown', {'cache_type': self.feature_store.cache_name,
                                                                                                    'store_id': self.feature_store.store_id,
                                                                                                    'executor_id': self.feature_store.executor_id,
                                                                                                    'num_stores': len(self.feature_store.caches),
                                                                                                    'executors_per_store': self.feature_store.executors_per_store})

                        clear_timers()
                        # TODO reset feature store state
                        self.feature_store.reset_cache()
                        print(f"Engine {self.device_id}: finished trial {cur_trial}")
                        print(f"Engine {self.device_id}: {self.feature_store.lock_conflicts} lock conflicts")
                        self.feature_store.lock_conflicts = 0
                        self.trial_barriers[cur_trial].wait()
                        cur_trial += 1
                        
        if use_prof and self.feature_store.executor_id == 0:
            prof.export_chrome_trace(f'multiprocess_trace_store_{self.device_id}_executor_{self.feature_store.executor_id}.json')

        export_dict_as_pd(self.feature_store.lock_conflict_trace, 'pipeline_trace', {'cache_type': self.feature_store.cache_name,
                                                                                    'store_id': self.feature_store.store_id,
                                                                                    'executor_id': self.feature_store.executor_id,
                                                                                    'num_stores': len(self.feature_store.caches),
                                                                                    'executors_per_store': self.feature_store.executors_per_store}, 0)
        self.finish_barrier.wait()


