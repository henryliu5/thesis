import torch
from torch.multiprocessing import Process, Queue, Barrier
from fast_inference.request_generator import Request, RequestType
from fast_inference.feat_server import FeatureServer, ManagedCacheServer
from fast_inference.sampler import InferenceSampler
from fast_inference.timer import Timer, enable_timers, clear_timers, print_timer_info
from torch.profiler import profile, record_function, ProfilerActivity
from contextlib import nullcontext

class InferenceEngine(Process):
    def __init__(self, request_queue: Queue,
                 response_queue: Queue,
                 start_barrier: Barrier,
                 finish_barrier: Barrier,
                 device: torch.device,
                 feature_store: FeatureServer,
                 logical_g,
                 model):
        
        super().__init__()
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.start_barrier = start_barrier
        self.finish_barrier = finish_barrier
        self.device = device
        self.device_id = device.index
        self.feature_store = feature_store

        self.logical_g = logical_g
        self.sampler = InferenceSampler(logical_g.to(self.device))
        self.model = model.to(device)

    def run(self):
        # torch.set_num_threads(16)

        # Need to re-pin the feature store buffer
        for k, v in self.feature_store.pinned_buf_dict.items():
            self.feature_store.pinned_buf_dict[k] = v.pin_memory()

        self.feature_store.init_counts(self.feature_store.num_nodes)
        if type(self.feature_store) == ManagedCacheServer:
            self.feature_store.start_manager()

        print('InferenceEngine', self.device_id, 'started')
        self.start_barrier.wait()

        update_window = 10
        requests_handled = 0
    
        use_prof = False
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) if use_prof else nullcontext() as prof:
            enable_timers()
            with torch.cuda.device(self.device): # needed to set timers on correct device
                while True:
                    req = self.request_queue.get()

                    if requests_handled % update_window == 0:
                        self.feature_store.compute_topk()
                        self.feature_store.update_cache(['feat'])

                    # print(req.nids.is_shared())
                    if req.req_type == RequestType.INFERENCE:
                        with Timer('exec request'):
                            mfgs = self.sampler.sample(req.nids, req.edges, use_gpu_sampling=True, device=self.device)

                            with Timer('dataloading', track_cuda=True):
                                required_feats = mfgs[0].ndata['_ID']['_N']
                                inputs, mfgs = self.feature_store.get_features(required_feats, feats=['feat'], mfgs=mfgs)
                                inputs = inputs['feat']
                            
                            with Timer('model'):
                                x = self.model(mfgs, inputs)
                                x.cpu()

                    requests_handled += 1

                    self.response_queue.put(Request(None, None, None, req.id, req.trial, RequestType.RESPONSE if req.req_type == RequestType.INFERENCE else req.req_type, req.start_time))
                    if req.req_type == RequestType.SHUTDOWN:
                        print(f'InferenceEngine {self.device_id} received shutdown request, id: {req.id}')
                        break
        if use_prof:
            prof.export_chrome_trace(f'multiprocess_trace_rank_{self.device_id}.json')

        print_timer_info()
        self.finish_barrier.wait()


