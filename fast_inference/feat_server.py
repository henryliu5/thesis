import torch
import dgl
from typing import List

class FeatureServer:
    def __init__(self, g: dgl.DGLGraph, device: torch.device or str):
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

    def get_features(self, node_ids: torch.LongTensor, feats: List[str]):
        """Get features for a list of nodes.

        Features are fetched from GPU memory if cached, otherwise from CPU memory.

        Args:
            node_ids (torch.Tensor): A 1-D tensor of node IDs.
            feats (List[str]): List of strings corresponding to feature keys that should be fetched.
        """
        res = {}
        # Used to mask this particular request - not to mask the cache!!
        gpu_mask = self.nid_is_on_gpu[node_ids]
        cpu_mask = ~gpu_mask

        for feat in feats:
            feat_shape = list(self.g.ndata[feat].shape[1:])
            # Create tensor with shape [number of nodes] x feature shape to hold result
            res_tensor = torch.zeros(
                tuple([node_ids.shape[0]] + feat_shape), device=self.device)

            # Start copy to GPU mem
            required_cpu_features = self.g.ndata[feat][node_ids[cpu_mask]]
            res_tensor[cpu_mask] = required_cpu_features.to(
                self.device, non_blocking=True)

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
