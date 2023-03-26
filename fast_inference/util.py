from torch.multiprocessing import Lock
import torch
import dgl
from fast_inference.feat_server import FeatureServer, CountingFeatServer, ManagedCacheServer
from typing import List

def create_feature_stores(cache_type: str, num_stores: int, graph: dgl.DGLGraph, track_feature_types: List[str], cache_percent: float,
                    use_pinned_mem: bool = True, profile_hit_rate: bool = False, pinned_buf_size: int = 150_000) -> List[FeatureServer]:
    """Create num_stores feature stores and configure for P2P communication

    Args:
        cache_type (str): _description_
        num_stores (int): _description_
        graph (dgl.DGLGraph): _description_
        track_feature_types (List[str]): _description_
        cache_percent (float): _description_
        use_pinned_mem (bool, optional): _description_. Defaults to True.
        profile_hit_rate (bool, optional): _description_. Defaults to False.
        pinned_buf_size (int, optional): _description_. Defaults to 150_000.

    Returns:
        List[FeatureServer]: _description_
    """
    num_nodes = graph.num_nodes()
    features = {f: graph.ndata[f] for f in track_feature_types}
    out_deg = graph.out_degrees()
    nids = torch.arange(num_nodes)

    additional_args = {}
    if cache_type == 'static':
        store_type = FeatureServer
    elif cache_type == 'count':
        store_type = CountingFeatServer
        additional_args['peer_lock'] = [Lock() for _ in range(num_stores)]
    elif cache_type == 'cpp':
        store_type = ManagedCacheServer
    elif cache_type == 'cpp_lock':
        store_type = ManagedCacheServer
        additional_args['use_locking'] = True

    feature_stores = []
    for device_id in range(num_stores):
        # mod partition best nodes across all GPUs
        part_nids = nids[nids % num_stores == device_id]
        _, indices = torch.topk(out_deg[part_nids], int(graph.num_nodes() * cache_percent / num_stores), sorted=True)
        part_indices = part_nids[indices]

        f = store_type(num_nodes, features, torch.device(
            'cuda', device_id), ['feat'], use_pinned_mem=use_pinned_mem, profile_hit_rate=profile_hit_rate, pinned_buf_size=pinned_buf_size, **additional_args)
        f.set_static_cache(part_indices, ['feat']) 
        feature_stores.append(f)

    del out_deg

    for device_id in range(num_stores):
        feature_stores[device_id].set_peer_group(feature_stores)
        feature_stores[device_id].init_counts(num_nodes)

    return feature_stores