import torch
import dgl
from typing import List
from fast_inference.timer import Timer

class FeatureServer:
    def __init__(self, g: dgl.DGLGraph, device: torch.device or str, profile_hit_rate = False):
        """ Initializes a new FeatureServer

        Args:
            g (dgl.DGLGraph): Graph whose features are to be served. Graph should be on CPU.
            device (torch.device): Device where feature server should store cache
        """
        assert (g.device == torch.device('cpu'))
        self.g = g
        self.device = device
        self.nid_is_on_gpu = torch.zeros(g.num_nodes()).bool()
        self.cache_mapping = - \
            torch.ones(g.num_nodes(), device=self.device).long()
        self.cache = {}
        self.profile = profile_hit_rate
        self.requests = 0
        self.cache_hits = 0

        # NOTE allocate "small" pinned buffers to place features that will be transferred
        # TODO turn this into a dictionary of feats and make work for any graph
        self.orig_pinned_feature_output = torch.empty((150_000, g.ndata['feat'].shape[1]), dtype=torch.float, pin_memory=True)

    def get_features(self, node_ids: torch.LongTensor, feats: List[str]):
        """Get features for a list of nodes.

        Features are fetched from GPU memory if cached, otherwise from CPU memory.

        Args:
            node_ids (torch.Tensor): A 1-D tensor of node IDs.
            feats (List[str]): List of strings corresponding to feature keys that should be fetched.
        """
        with Timer('get_features()'):
            res = {}

            # Used to mask this particular request - not to mask the cache!!
            with Timer('compute gpu/cpu mask'):
                gpu_mask = self.nid_is_on_gpu[node_ids]
                cpu_mask = ~gpu_mask

            if self.profile:
                self.requests += node_ids.shape[0]
                self.cache_hits += gpu_mask.long().sum()

            for feat in feats:
                feat_shape = list(self.g.ndata[feat].shape[1:])
                with Timer('allocate res tensor', track_cuda = True):
                    # Create tensor with shape [number of nodes] x feature shape to hold result
                    res_tensor = torch.zeros(
                        tuple([node_ids.shape[0]] + feat_shape), device=self.device)

                # Start copy to GPU mem
                with Timer('mask cpu feats'):
                    m = node_ids[cpu_mask]
                    # Perform resizing if necessary
                    if m.shape[0] > self.orig_pinned_feature_output.shape[0]:
                        self.orig_pinned_feature_output = self.orig_pinned_feature_output.resize_((m.shape[0], self.orig_pinned_feature_output.shape[1]))
                    required_cpu_features = self.orig_pinned_feature_output.narrow(0, 0, m.shape[0])
                with Timer('index feats'):
                    # print('feat shape', self.feats.shape, 'device', self.feats.device, self.feats.is_contiguous())
                    # print('m shape', m.shape, 'device', m.device, m.is_contiguous())
                    # print('type self.g.ndata', type(self.feats))
                    # print('type', type(self.g.ndata[feat]))
                    # s = time.time()


                    torch.index_select(self.g.ndata[feat], 0, m, out=required_cpu_features)
                    # print(self.feats.shape, self.feats, self.feats.device)
                    # print(m.shape, m, m.device)
                    # required_cpu_features = self.feats[m]
                    # required_cpu_features = torch.index_select(self.feats, 0, m)
                with Timer('actual copy', track_cuda=True):
                    res_tensor[cpu_mask] = required_cpu_features.to(
                        self.device, non_blocking=False)

                with Timer('move cached features', track_cuda=True):
                    # Features from GPU mem
                    # self.cache_mapping maps the global node id to the respective index in the cache
                    if feat in self.cache: # hacky drop in for torch.any(gpu_mask)
                        required_gpu_features = self.cache[feat][self.cache_mapping[node_ids[gpu_mask]]]
                        res_tensor[gpu_mask] = required_gpu_features
                
                res[feat] = res_tensor

        return res

    def set_static_cache(self, node_ids: torch.Tensor, feats: List[str]):
        """Define a static cache using the given node ids.

        Args:
            node_ids (torch.Tensor): Elements should be node ids whose features are to be cached in GPU memory.
            feats (List[str]): List of strings corresponding to feature keys that should be cached.
        """
        self.cache_size = node_ids.shape[0]
        self.nid_is_on_gpu[node_ids] = True
        self.cache_mapping[node_ids] = torch.arange(
            self.cache_size, device=self.device)

        for feat in feats:
            self.cache[feat] = self.g.ndata[feat][node_ids].to(self.device)

    def get_cache_hit_ratio(self):
        assert (self.profile), "Profiling must be turned on for this FeatureServer."
        assert (self.requests != 0), "No requests received by FeatureServer yet."
        return (self.cache_hits / self.requests).item()