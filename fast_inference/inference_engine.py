import torch
from torch.multiprocessing import Process, Queue, Barrier
from fast_inference.request_generator import Request, RequestType
from fast_inference.feat_server import FeatureServer
from fast_inference.dataset import InferenceDataset
from fast_inference.sampler import InferenceSampler
from tqdm import tqdm
import dgl

class InferenceEngine(Process):
    def __init__(self, request_queue: Queue,
                 response_queue: Queue,
                 start_barrier: Barrier,
                 device: torch.device,
                 feature_store: FeatureServer,
                 logical_g,
                 model):
        
        super().__init__()
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.start_barrier = start_barrier
        self.device = device
        self.device_id = device.index
        self.feature_store = feature_store

        self.logical_g = logical_g
        self.sampler = InferenceSampler(logical_g.to(self.device))
        self.model = model.to(device)

    def run(self):
        # torch.set_num_threads(16)
        print('InferenceEngine', self.device_id, 'started')
        self.start_barrier.wait()

        # infer_data = InferenceDataset('ogbn-products', 0.1, partitions=5, force_reload=False, verbose=True)
        # self.trace = infer_data.create_inference_trace(subgraph_bias=None)
        # g = infer_data[0]
        # logical_g = dgl.graph(g.edges())

        # sampler = InferenceSampler(logical_g.to(self.device))

        # self.trials = 1
        # self.batch_size = 256
        # self.max_iters = 256000

        # n = len(self.trace)
        # for trial in range(self.trials):
        #     print('Starting trial', trial)
        #     for i in tqdm(range(0, min(n, self.max_iters * self.batch_size), self.batch_size)):
        #         if i + self.batch_size >= n:
        #             continue
        #         nids = self.trace.nids[i:i+self.batch_size]
        #         features = self.trace.features[i:i+self.batch_size]
        #         edges = self.trace.edges[i:i+self.batch_size]

        #         mfgs = sampler.sample(nids, edges, use_gpu_sampling=True)

        #         required_feats = mfgs[0].ndata['_ID']['_N']
        #         inputs, mfgs = self.feature_store.get_features(required_feats, feats=['feat'], mfgs=mfgs)
        #         inputs = inputs['feat']
        while True:
            req = self.request_queue.get()
            # print(req.nids.is_shared())
            if req.req_type == RequestType.INFERENCE:
                mfgs = self.sampler.sample(req.nids, req.edges, use_gpu_sampling=True, device=self.device)

            self.response_queue.put(Request(None, None, None, req.id, req.trial, RequestType.RESPONSE if req.req_type == RequestType.INFERENCE else req.req_type, req.start_time))
            if req.req_type == RequestType.SHUTDOWN:
                exit(f'Received shutdown request, id: {req.id}')


