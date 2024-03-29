import torch
import dgl
from fast_inference.feat_server import FeatureServer, CountingFeatServer, ManagedCacheServer, LFUServer, FrequencySynchronousCache
from fast_inference_cpp import shm_setup
from fast_inference.device_cache import DeviceFeatureCache
from typing import List

def create_feature_stores(cache_type: str, num_stores: int, executors_per_store: int, graph: dgl.DGLGraph, track_feature_types: List[str], cache_percent: float,
                    use_pinned_mem: bool = True, profile_hit_rate: bool = False, pinned_buf_size: int = 150_000, use_pytorch_direct: bool = False) -> List[FeatureServer]:
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
    shm_setup(num_stores, executors_per_store)

    num_nodes = graph.num_nodes()
    features = {f: graph.ndata[f] for f in track_feature_types}
    out_deg = graph.out_degrees()
    nids = torch.arange(num_nodes)

    additional_args = {}
    additional_args['executors_per_store'] = executors_per_store
    additional_args['total_stores'] = num_stores
    if cache_type == 'static':
        store_type = FeatureServer
    elif cache_type == 'count':
        store_type = CountingFeatServer
        # from fast_inference.rwlock import RWLock
        # from torch.multiprocessing import Lock
        # if num_stores == 1 and executors_per_store == 1:
        #     additional_args['peer_lock'] = None
        # else:
        #     additional_args['peer_lock'] = [RWLock() for _ in range(num_stores)]
        # additional_args['peer_lock'] = [RWLock() for _ in range(num_stores)]
    elif cache_type == 'cpp':
        store_type = ManagedCacheServer
        additional_args['use_locking'] = False
    elif cache_type == 'cpp_lock':
        store_type = ManagedCacheServer
        additional_args['use_locking'] = True
    elif cache_type == 'lfu':
        store_type = LFUServer
    elif cache_type == 'freq-sync':
        store_type = FrequencySynchronousCache
    else:
        print('Cache type', cache_type, 'not supported')
        exit()

    caches = []
    for store_id in range(num_stores):
        device_id = store_id % torch.cuda.device_count()
        part_nids = nids[nids % num_stores == device_id]
        _, indices = torch.topk(out_deg[part_nids], int(graph.num_nodes() * cache_percent / num_stores), sorted=True)
        part_indices = part_nids[indices]
        print('Caching', part_indices.shape[0], 'nodes')
        cache = DeviceFeatureCache.initialize_cache(init_nids=part_indices.to(torch.device('cuda', device_id)), 
                                                    num_nodes=num_nodes,
                                                    feats=features,
                                                    device=torch.device('cuda', device_id),
                                                    cache_id=store_id,
                                                    total_caches=num_stores)
        caches.append(cache)

    feature_stores = []
    for store_id in range(num_stores):
        device_id = store_id % torch.cuda.device_count()
        executors = []
        for executor_id in range(executors_per_store):
            if executor_id == 0:
                additional_args['is_leader'] = True
            else:
                additional_args['is_leader'] = False

            f = store_type(caches, num_nodes, features, torch.device(
                'cuda', device_id), store_id, executor_id, ['feat'], use_pinned_mem=use_pinned_mem, profile_hit_rate=profile_hit_rate, pinned_buf_size=pinned_buf_size, use_pytorch_direct=use_pytorch_direct, **additional_args)
            
            # if executor_id == 0:
            #     # mod partition best nodes across all GPUs
            #     f.set_static_cache(part_indices.to(torch.device('cuda', device_id)), ['feat']) 
            # elif store_type == ManagedCacheServer:
            #     f.set_shared_cache(executors[0].original_cache_indices, executors[0].cache_size, executors[0].nid_is_on_gpu, executors[0].cache_mapping, executors[0].cache, executors[0].is_cache_candidate)
            # else:
            #     f.set_shared_cache(executors[0].original_cache_indices, executors[0].cache_size, executors[0].nid_is_on_gpu, executors[0].cache_mapping, executors[0].cache)
            
            executors.append(f)
            
        feature_stores.append(executors)

    del out_deg

    for device_id in range(num_stores):
        for executor_id in range(executors_per_store):
            # feature_stores[device_id][executor_id].set_peer_group([stores[0] for stores in feature_stores])
            feature_stores[device_id][executor_id].init_counts(num_nodes)

    return feature_stores