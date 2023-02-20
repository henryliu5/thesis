import torch
import dgl
from typing import List, Optional
from fast_inference.timer import Timer

class FeatureServer:
    def __init__(self, g: dgl.DGLGraph, device: torch.device or str, track_features: List[str], profile_hit_rate: bool = False, pinned_buf_size: int = 150_000):
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

        # Pinned memory buffers for placing gathered CPU features prior to CPU->GPU copy
        self.pinned_buf_dict = {}
        # NOTE allocate "small" pinned buffers to place features that will be transferred
        for feature in track_features:
            # TODO make these buffers work with features that are not 1D (see pytest test)
            self.pinned_buf_dict[feature] = torch.empty((pinned_buf_size, g.ndata[feature].shape[1]), dtype=torch.float, pin_memory=True)

    def get_features(self, node_ids: torch.LongTensor, feats: List[str], mfgs: Optional[dgl.DGLGraph]=None):
        """Get features for a list of nodes.

        Features are fetched from GPU memory if cached, otherwise from CPU memory.

        Args:
            node_ids (torch.Tensor): A 1-D tensor of node IDs.
            feats (List[str]): List of strings corresponding to feature keys that should be fetched.
        """
        if mfgs is None:
            mfgs = []

        with Timer('get_features()'):
            node_ids = node_ids.cpu()
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
                    if m.shape[0] > self.pinned_buf_dict[feat].shape[0]:
                        self.pinned_buf_dict[feat] = self.pinned_buf_dict[feat].resize_((m.shape[0], self.pinned_buf_dict[feat].shape[1]))
                    required_cpu_features = self.pinned_buf_dict[feat].narrow(0, 0, m.shape[0])
                with Timer('feature gather'):
                    torch.index_select(self.g.ndata[feat], 0, m, out=required_cpu_features)
                    # NOTE Can remove above for "slow mode"
                    # required_cpu_features = torch.index_select(self.g.ndata[feat], 0, m)

                with Timer('CPU-GPU copy', track_cuda=True):
                    # Copy CPU features
                    res_tensor[cpu_mask] = required_cpu_features.to(
                        self.device, non_blocking=True)
                    # Copy MFGs
                    mfgs = [mfg.to(self.device) for mfg in mfgs]

                with Timer('move cached features', track_cuda=True):
                    # Features from GPU mem
                    # self.cache_mapping maps the global node id to the respective index in the cache
                    if feat in self.cache: # hacky drop in for torch.any(gpu_mask)
                        required_gpu_features = self.cache[feat][self.cache_mapping[node_ids[gpu_mask]]]
                        res_tensor[gpu_mask] = required_gpu_features
                
                res[feat] = res_tensor

        return res, mfgs

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