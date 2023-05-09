from dataclasses import dataclass
from typing import Dict
import torch

@dataclass(frozen=True)
class DeviceFeatureCache:
    '''
    Feature cache hosted on a device.

    This is a frozen dataclass so it is easy to share amongst processes
    with a guarantee* that the same GPU memory buffers are being operated on
    by different processes.
    '''
    cache_size: int
    cache_mask: torch.Tensor
    cache_mapping: torch.Tensor
    reverse_mapping: torch.Tensor
    cache: Dict[str, torch.Tensor]

    is_cache_candidate: torch.Tensor

    device: torch.device

    '''
    Caches have an id, where a cache will hold all nids s.t. nid % total_caches == cache_id
    #TODO add check to make sure this is always the case
    '''
    cache_id: int
    total_caches: int
    
    @staticmethod
    def initialize_cache(init_nids: torch.Tensor, num_nodes: int, feats: Dict[str, torch.Tensor], device: torch.device, cache_id: int, total_caches: int):
        # if self.original_cache_indices is None:
        #     self.original_cache_indices = init_nids

        cache_size = init_nids.shape[0]
        cache_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        cache_mapping = -1 * torch.ones(num_nodes, device=device, dtype=torch.long)
        reverse_mapping = init_nids
        assert(init_nids.device == device)

        cache_mask[init_nids] = True
        cache_mapping[init_nids] = torch.arange(cache_size, device=device)

        cache = {}
        for feat in feats:
            cache[feat] = feats[feat][init_nids.cpu()].to(device)

        is_cache_candidate = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        return DeviceFeatureCache(cache_size, cache_mask, cache_mapping, reverse_mapping, cache, is_cache_candidate, device, cache_id, total_caches)