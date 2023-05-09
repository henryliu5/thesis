import torch
import dgl
from typing import List, Optional, Dict
from fast_inference.timer import Timer, export_dict_as_pd, TRACES
from fast_inference_cpp import CacheManager, SHMLocks
from fast_inference.device_cache import DeviceFeatureCache
import time
import functools

class FeatureServer:
    cache_name = 'static'
    def __init__(self, 
                 caches: List[DeviceFeatureCache],
                 num_nodes: int,
                 features: Dict[str, torch.Tensor], 
                 device: torch.device or str,
                 store_id: int,
                 executor_id: int,
                 track_features: List[str],
                 use_pinned_mem: bool = True,
                 profile_hit_rate: bool = False,
                 pinned_buf_size: int = 150_000,
                 peer_lock = None,
                 use_locking = False,
                 is_leader: bool = True,
                 total_stores: int = -1,
                 executors_per_store: int = -1,
                 use_pytorch_direct: bool = False):
        """ Initializes a new FeatureServer

        Args:
            g (dgl.DGLGraph): Graph whose features are to be served. Graph should be on CPU.
            device (torch.device): Device where feature server should store cache
        """
        # assert (g.device == torch.device('cpu'))
        self.features = features
        self.num_nodes = num_nodes
        self.device = device
        self.device_index = device.index
        self.store_id = store_id
        self.executor_id = executor_id

        self.use_pinned_mem = use_pinned_mem
        self.profile = profile_hit_rate
        self.profile_info = {'request_size': [], 'cache_hits': [], 'hit_rate': []}

        # Pinned memory buffers for placing gathered CPU features prior to CPU->GPU copy
        self.pinned_buf_dict = {}
        # NOTE allocate "small" pinned buffers to place features that will be transferred
        for feature in track_features:
            # TODO make these buffers work with features that are not 1D (see pytest test)
            self.pinned_buf_dict[feature] = torch.empty((pinned_buf_size, features[feature].shape[1]), dtype=torch.float, pin_memory=True)

        self.original_cache_indices = None

        self.caches = caches
        self.peer_streams = None

        self.peer_lock = peer_lock
        self.lock_conflicts = 0
        self.lock_conflict_trace = {'wait_time': []}
                                    # 'executor_id': [], 'store_id': [], 'num_stores': [], 'executors_per_store': []}
        self.use_locking = use_locking
        self.is_leader = is_leader
        self.total_stores = total_stores
        self.executors_per_store = executors_per_store
        self.requests_handled = 0

        self.use_pytorch_direct = use_pytorch_direct

    def get_peer_features(self, node_ids: torch.LongTensor, feat: str):
        if self.peer_streams is None:
            self.peer_streams = [torch.cuda.Stream(device=self.device) for _ in self.caches]
        assert(node_ids.device == self.device), f'node ids {node_ids.device}, self.device {self.device}'
        
        result_masks = []
        result_features = []

        dur = 0 
        num_peers = len(self.caches)
        for i in range(num_peers):
            peer = self.caches[i]

            with torch.cuda.stream(self.peer_streams[i]):
                s = time.perf_counter()
                self.sync_cache_read_start(i)
                dur += time.perf_counter() - s

                # Only transfer node ids that belong to that GPU
                # peer_mask = gpu_mask & (node_ids % num_peers == i)
                #! TODO figure out a way to possibly not transfer all nids (makes masking weird because dim 0 is reduced)
                node_ids_peer = node_ids.to(peer.device)
                peer_mask = peer.cache_mask[node_ids_peer]
                peer_nids = node_ids_peer[peer_mask]

                mapping = peer.cache_mapping[peer_nids]
                
                if len(self.caches) > 1:
                    assert(peer.cache_mask.is_shared())
                    assert(peer.cache_mapping.is_shared())
                    assert(peer.cache[feat].is_shared())

                assert(peer.cache_mask.long().sum() <= peer.cache_size)

                # assert(torch.all(mapping >= 0))
                result_features.append(peer.cache[feat][mapping].to(self.device))
                peer_mask = peer_mask.to(self.device)

                self.sync_cache_read_end(i)

            result_masks.append(peer_mask)

        [stream.synchronize() for stream in self.peer_streams]

        if dur >= 0.001:
            print('Waited for lock', dur)
            self.lock_conflicts += 1

        peer_lock_trace_name = 'acquire peer lock'
        if peer_lock_trace_name not in TRACES:
            TRACES[peer_lock_trace_name] = []
        TRACES[peer_lock_trace_name].append(dur)
        self.lock_conflict_trace['wait_time'].append(dur)
        # self.lock_conflict_trace['executor_id'].append(self.executor_id)
        # self.lock_conflict_trace['store_id'].append(self.store_id)
        # self.lock_conflict_trace['num_stores'].append(self.total_stores)
        # self.lock_conflict_trace['executors_per_store'].append(self.executors_per_store)

        return result_masks, result_features

    def _get_features(self, node_ids: torch.LongTensor, feats: List[str], mfgs: Optional[dgl.DGLGraph]=None, request_id: Optional[int]=None):
        """Get features for a list of nodes.

        Features are fetched from GPU memory if cached, otherwise from CPU memory.

        Args:
            node_ids (torch.Tensor): A 1-D tensor of node IDs.
            feats (List[str]): List of strings corresponding to feature keys that should be fetched.
        """
        assert(node_ids.device != torch.device('cpu'))
        if request_id != None:
            if 'request_id' not in self.lock_conflict_trace:
                self.lock_conflict_trace['request_id'] = []
            self.lock_conflict_trace['request_id'].append(request_id)

        if mfgs is None:
            mfgs = []

        with Timer('get_features()'):
            res = {}

            with Timer('get peer features'):
                for feat in feats:
                    peer_masks, gpu_features = self.get_peer_features(node_ids, feat)

            # Used to mask this particular request - not to mask the cache!!
            with Timer('compute gpu/cpu mask'):
                torch.cuda.current_stream().synchronize()
                # gpu_mask = self.nid_is_on_gpu[node_ids]
                gpu_mask = functools.reduce(torch.logical_or, peer_masks)

                if len(self.caches) > 1:
                    # Verify isolation between GPU cachces
                    assert(not torch.any(functools.reduce(torch.logical_and, peer_masks)))
                
                cpu_mask = ~gpu_mask

            if self.profile:
                # TODO add statistics for each cache
                self.profile_info['request_size'].append(node_ids.shape[0])
                cache_hits = gpu_mask.int().sum().item()
                self.profile_info['cache_hits'].append(cache_hits)
                self.profile_info['hit_rate'].append(cache_hits / node_ids.shape[0])

            for feat in feats:
                feat_shape = list(self.features[feat].shape[1:])
                with Timer('allocate res tensor', track_cuda = True):
                    # Create tensor with shape [number of nodes] x feature shape to hold result
                    res_tensor = torch.zeros(
                        tuple([node_ids.shape[0]] + feat_shape), device=self.device)

                # Start copy to GPU mem
                with Timer('mask cpu feats'):
                    m = node_ids[cpu_mask]
                    # Perform resizing if necessary
                    if self.use_pinned_mem:
                        if m.shape[0] > self.pinned_buf_dict[feat].shape[0]:
                            self.pinned_buf_dict[feat] = self.pinned_buf_dict[feat].resize_((m.shape[0], self.pinned_buf_dict[feat].shape[1]))
                        required_cpu_features = self.pinned_buf_dict[feat].narrow(0, 0, m.shape[0])

                with Timer('feature gather'):
                    if self.use_pytorch_direct:
                        if not self.features[feat].is_pinned():
                            # print(type(self.features[feat]))
                            self.features[feat] = self.features[feat].pin_memory()
                            # self.features[feat] = dgl.utils.pin_memory_inplace(self.features[feat])
                        required_cpu_features = dgl.utils.pin_memory.gather_pinned_tensor_rows(self.features[feat], m.to(self.device))
                    else:
                        if self.use_pinned_mem:
                            # Places indices directly into pinned memory buffer
                            torch.index_select(self.features[feat], 0, m.cpu(), out=required_cpu_features)
                        else:
                            #"slow mode"
                            required_cpu_features = torch.index_select(self.features[feat], 0, m)

                with Timer('CPU-GPU copy', track_cuda=True):
                    # Copy CPU features
                    cpu_feats = required_cpu_features.to(
                        self.device, non_blocking=True)
                    res_tensor[cpu_mask] = cpu_feats
                    # Copy MFGs
                    mfgs = [mfg.to(self.device) for mfg in mfgs]

                with Timer('move cached features', track_cuda=True):
                    # Features from GPU mem
                    if len(self.caches) == 1:
                        res_tensor[gpu_mask] = gpu_features[0]
                    else:
                        for i in range(len(self.caches)):
                            res_tensor[peer_masks[i]] = gpu_features[i]
                    # # self.cache_mapping maps the global node id to the respective index in the cache
                    # mapping = self.cache_mapping[node_ids[gpu_mask]]
                    # assert(torch.all(mapping >= 0))
                    # required_gpu_features = self.cache[feat][mapping]
                    # res_tensor[gpu_mask] = required_gpu_features
                
                res[feat] = res_tensor

        return res, mfgs, m, cpu_feats
    
    def get_features(self, node_ids: torch.LongTensor, feats: List[str], mfgs: Optional[dgl.DGLGraph]=None):
        res_tensor_dict, mfgs, newly_transferred_nids, newly_transferred_feats = self._get_features(node_ids, feats, mfgs)

        return res_tensor_dict, mfgs

    def export_profile(self, path, current_config):
        if self.profile:
            export_dict_as_pd(self.profile_info, path, current_config)
        else:
            print('FeatureServer.export_profile called but profiling disabled!')

    def init_counts(self, *args):
        pass

    def update_cache(self, *args):
        pass

    def sync_cache_read_start(self, index: int):
        """Perform necessary synchronization to begin reading consistent cache state

        Args:
            index (int): Device index to be read from
        """
        pass

    def sync_cache_read_end(self, index: int):
        """Releease relevant synchronization resources related to self.sync_cache_read_start

        Args:
            index (int): Device index to be read from
        """
        pass

    def reset_cache(self, *args):
        for k in self.profile_info.keys():
            self.profile_info[k] = []
        for k in self.lock_conflict_trace.keys():
            self.lock_conflict_trace[k] = []

class CountingFeatServer(FeatureServer):
    cache_name = 'count'
    # TODO tidy this up, no need to num total nodes again here
    def init_counts(self, num_total_nodes):
        self.counts = torch.zeros(num_total_nodes, dtype=torch.bfloat16, device=self.device)
        self.count_stream = None

        self.most_common_nids = None
        self.topk_started = False
        print('FeatureStore', self.device_index, 'initialized counts')

    def init_locks(self):
        self.locks = SHMLocks()

    def update_cache(self):
        if not self.is_leader:
            return

        cache = self.caches[self.store_id].cache
        cache_mask = self.caches[self.store_id].cache_mask
        cache_mapping = self.caches[self.store_id].cache_mapping
        cache_size = self.caches[self.store_id].cache_size
        # assert(self.nid_is_on_gpu.is_shared())
        # assert(self.cache_mapping.is_shared())

        self.locks.write_lock(self.device_index)

        # write_lock(self.device_index)
        # if not self.peer_lock is None:
        #     if hasattr(self.peer_lock[self.device_index], 'acquire'):
        #         self.peer_lock[self.device_index].acquire()
        #     else:
        #         self.peer_lock[self.device_index].writer_lock.acquire()
            # self.peer_lock[self.device_index].writer_lock.acquire()
        [torch.cuda.synchronize(i) for i in range(torch.cuda.device_count())]

        if len(self.caches) > 1:
            # total_counts = functools.reduce(torch.add, [peer.counts.to(self.device) for peer in self.caches])
            # big_graph_arange = torch.arange(self.num_nodes, device=self.device)
            # part_nids = big_graph_arange[big_graph_arange % len(self.caches) == self.device_index]
            # _, most_common_idxs = torch.topk(total_counts[part_nids], self.cache_size, sorted=False)
            # most_common_nids = part_nids[most_common_idxs]
            part_counts = self.counts[self.device_index::len(self.caches)]
            _, top_part_idxs = torch.topk(part_counts, cache_size, sorted=True)
            most_common_nids = top_part_idxs * len(self.caches) + self.device_index
        else:
            v, most_common_nids = torch.topk(self.counts, cache_size, sorted=True)
            
        assert(torch.all(self.counts >= 0))
        print(most_common_nids)
        print(v[:10])

        cache_mask_device = cache_mask
        most_common_mask = torch.zeros(self.num_nodes, device=self.device).bool()
        most_common_mask[most_common_nids] = True

        # Mask for node ids that need features to be transferred
        # (new entrants to cache)            
        requires_update_mask = torch.logical_and(most_common_mask, torch.logical_not(cache_mask_device))

        # Indices of who can be replaced in the cache
        replace_nids_mask = torch.logical_and(~most_common_mask, cache_mask_device)
        requires_update_cache_idx = cache_mapping[replace_nids_mask]

        if requires_update_cache_idx.shape[0] != 0:
            for feat in cache.keys():
                old_shape = cache[feat].shape
                cache[feat][requires_update_cache_idx] = self.features[feat][requires_update_mask.cpu()].to(self.device)
                assert(cache[feat].shape == old_shape)

            cache_mapping[requires_update_mask] = requires_update_cache_idx

            cache_mask[:] = most_common_mask
            cache_mapping[~most_common_mask] = -1

            [torch.cuda.synchronize(i) for i in range(torch.cuda.device_count())]
            
        self.locks.write_unlock(self.device_index)
        # write_unlock(self.device_index)
        # if not self.peer_lock is None:
        #     if hasattr(self.peer_lock[self.device_index], 'release'):
        #         self.peer_lock[self.device_index].release()
        #     else:
        #         self.peer_lock[self.device_index].reader_lock.release()
            # self.peer_lock[self.device_index].reader_lock.release()
        # torch.div(self.counts, 2, rounding_mode='floor', out=self.counts)
        self.counts /= 2
        torch.cuda.synchronize(self.device)

    def get_features(self, node_ids: torch.LongTensor, feats: List[str], mfgs: Optional[dgl.DGLGraph]=None):
        """Get features for a list of nodes.

        Features are fetched from GPU memory if cached, otherwise from CPU memory.

        Args:
            node_ids (torch.Tensor): A 1-D tensor of node IDs.
            feats (List[str]): List of strings corresponding to feature keys that should be fetched.
        """
        with Timer('update counts'):
            if self.is_leader:
                if self.count_stream is None:
                    self.count_stream = torch.cuda.Stream(device=self.device)
                with torch.cuda.stream(self.count_stream):
                    self.counts[node_ids] += 1

        return super().get_features(node_ids, feats, mfgs)
    
    def reset_cache(self):
        super().reset_cache()
        if self.is_leader:
            self.counts *= 0
        # self.set_static_cache(self.original_cache_indices, list(self.cache.keys()))

    def sync_cache_read_start(self, index: int):
        """Perform necessary synchronization to begin reading consistent cache state

        Args:
            index (int): Device index to be read from
        """
        # [torch.cuda.synchronize(i) for i in range(torch.cuda.device_count())]
        self.locks.read_lock(index)
        # read_lock(index)
        # if not self.peer_lock is None:
            # if hasattr(self.peer_lock[index], 'acquire'):
            #     self.peer_lock[index].acquire()
            # else:
            #     self.peer_lock[index].reader_lock.acquire()
            # self.peer_lock[index].reader_lock.acquire()
    def sync_cache_read_end(self, index: int):
        """Releease relevant synchronization resources related to self.sync_cache_read_start

        Args:
            index (int): Device index to be read from
        """
        # [torch.cuda.synchronize(i) for i in range(torch.cuda.device_count())]
        self.locks.read_unlock(index)
        # read_unlock(index)
        # if not self.peer_lock is None:
        #     if hasattr(self.peer_lock[index], 'release'):
        #         self.peer_lock[index].release()
        #     else:
        #         self.peer_lock[index].reader_lock.release()
            # self.peer_lock[index].reader_lock.release()
    
class LFUServer(CountingFeatServer):

    def update_cache(self, *args):
        torch.div(self.counts, 2, rounding_mode='floor', out=self.counts)

    def get_features(self, node_ids: torch.LongTensor, feats: List[str], mfgs: Optional[dgl.DGLGraph]=None):
        """Get features for a list of nodes.

        Features are fetched from GPU memory if cached, otherwise from CPU memory.

        Args:
            node_ids (torch.Tensor): A 1-D tensor of node IDs.
            feats (List[str]): List of strings corresponding to feature keys that should be fetched.
        """

        cache = self.caches[self.store_id].cache
        cache_mask = self.caches[self.store_id].cache_mask
        cache_mapping = self.caches[self.store_id].cache_mapping
        cache_size = self.caches[self.store_id].cache_size

        cpu_node_ids = node_ids.cpu()

        with Timer('super get features'):
            res_dict, res_mfg = super().get_features(node_ids, feats, mfgs)

        with Timer('LFU update'):
            # Perform LFU update
            # Admission policy is simply allow everything in
            gpu_mask = cache_mask[node_ids]
            cpu_mask = ~gpu_mask

            # Will want to add cache misses
            nids_to_add = node_ids[cpu_mask]

            for feat in feats:
                if nids_to_add.shape[0] > cache_size:
                    # Truncate if necessary, just take whatever first firts
                    nids_to_add = nids_to_add[:cache_size]

                count_of_cache_residents = self.counts[cache_mask]
                resident_mapping = torch.arange(cache_mask.shape[0], device=self.device)[cache_mask]
                # Replace lowest count
                _, replace_residents = torch.topk(count_of_cache_residents, k=nids_to_add.shape[0], largest=False, sorted=True)
                replace_nids = resident_mapping[replace_residents]

                cache_slots = cache_mapping[replace_nids]
                cache_mask[replace_nids] = False
                cache_mapping[replace_nids] = -1

                cache_mask[nids_to_add] = True
                cache_mapping[nids_to_add] = cache_slots

                old_shape = cache[feat].shape
                
                # Recall the above truncation - the features we want will be at the front of the result tensor
                cache[feat][cache_slots] = res_dict[feat][cpu_mask][:cache_size]
                assert(cache[feat].shape == old_shape)

        return res_dict, res_mfg


class ManagedCacheServer(FeatureServer):
    cache_name = 'cpp'
    def init_counts(self, num_total_nodes):
        self.num_nodes = num_total_nodes
        if self.is_leader:
            self.counts = torch.zeros(num_total_nodes, dtype=torch.bfloat16, device=self.device)

        self.topk_stream = None
        self.update_stream = None
        self.count_stream = None

        self.most_common_nids = None
        self.topk_started = False
        self.topk_processed = False

        self.is_cache_candidate = self.caches[self.store_id].is_cache_candidate
        if self.use_locking:
            self.cache_name = 'cpp_lock'

    def start_manager(self):
        cache = self.caches[self.store_id].cache
        cache_mask = self.caches[self.store_id].cache_mask
        cache_mapping = self.caches[self.store_id].cache_mapping
        reverse_mapping = self.caches[self.store_id].reverse_mapping
        cache_size = self.caches[self.store_id].cache_size
        for feat in cache:
            self.cache_manager = CacheManager(self.num_nodes, cache_size, self.device_index, len(self.caches), True, self.use_locking, self.total_stores, self.executors_per_store, self.executor_id)
            self.cache_manager.set_cache(self.features[feat], cache_mask, cache_mapping, reverse_mapping, cache[feat])
            self.cache_manager.set_cache_candidates(self.is_cache_candidate)
            break

    def update_cache(self):
        if not self.is_leader:
            return

        cache_size = self.caches[self.store_id].cache_size

        if self.topk_stream is None:
            self.topk_stream = torch.cuda.Stream(device=self.device)
        if self.update_stream is None:
            self.update_stream = torch.cuda.Stream(device=self.device)
    
        with torch.cuda.stream(self.topk_stream):
            torch.zeros(self.num_nodes, out=self.is_cache_candidate, dtype=torch.bool, device=self.device)

            if len(self.caches) > 1:
                # total_counts = functools.reduce(torch.add, [peer.counts.to(self.device) for peer in self.caches])
                # big_graph_arange = torch.arange(self.num_nodes, device=self.device)
                # part_nids = big_graph_arange[big_graph_arange % len(self.caches) == self.device_index]
                # _, most_common_idxs = torch.topk(total_counts[part_nids], self.cache_size, sorted=False)
                # most_common_nids = part_nids[most_common_idxs]
                part_counts = self.counts[self.device_index::len(self.caches)]
                _, top_part_idxs = torch.topk(part_counts, cache_size, sorted=False)
                self.most_common_nids = top_part_idxs * len(self.caches) + self.device_index
            else:
                _, self.most_common_nids = torch.topk(self.counts.to(self.device, non_blocking=True), cache_size, sorted=False)
                self.is_cache_candidate[self.most_common_nids] = True
                # self.topk_started = True
                # self.topk_processed = False
            # torch.div(self.counts, 2, rounding_mode='floor', out=self.counts)
            self.counts /= 2

    def get_features(self, node_ids: torch.LongTensor, feats: List[str], mfgs: Optional[dgl.DGLGraph]=None):
        """Get features for a list of nodes.

        Features are fetched from GPU memory if cached, otherwise from CPU memory.

        Args:
            node_ids (torch.Tensor): A 1-D tensor of node IDs.
            feats (List[str]): List of strings corresponding to feature keys that should be fetched.
        """
        gpu_nids = node_ids
        with Timer('update counts'):
            if self.is_leader:
                if self.count_stream is None:
                    self.count_stream = torch.cuda.Stream(device=self.device)
                with torch.cuda.stream(self.count_stream):
                    self.counts[node_ids] += 1

        res_tensor_dict, mfgs, newly_transferred_nids, newly_transferred_feats = self._get_features(node_ids, feats, mfgs)

        with Timer('cache update'):
                # with Timer('place in queue'):
                # !! WARNING: Must use "m" here!! Since the node ids and mask are on GPU, the CPU node id tensor
                # !! must be fully materialized by the time the tensor is placed on the queue
            # if self.is_leader:
            if self.topk_stream is not None:
                torch.cuda.current_stream().wait_stream(self.topk_stream)
            self.cache_manager.place_feats_in_queue(newly_transferred_feats, newly_transferred_nids)

        return res_tensor_dict, mfgs

    def reset_cache(self):
        super().reset_cache()
        if self.is_leader:
            self.counts *= 0
        # if self.is_leader:
        #     self.set_static_cache(self.original_cache_indices, list(self.cache.keys()))
        # for feat in self.cache:
        #     self.cache_manager.set_cache(self.features[feat], self.nid_is_on_gpu, 
        #                         self.cache_mapping, self.reverse_mapping.to(self.device), self.cache[feat])

    def sync_cache_read_start(self, index: int):
        """Perform necessary synchronization to begin reading consistent cache state

        Args:
            index (int): Device index to be read from
        """
        # with Timer('read lock enter'):
            # self.cache_manager.thread_enter()
        if self.use_locking:
            # [torch.cuda.synchronize(i) for i in range(torch.cuda.device_count())]
            self.cache_manager.read_lock(index)
        else:
            self.cache_manager.thread_enter(index, self.device_index, self.executor_id)
            # TODO put this actual isolation check everywhere
            # assert(not torch.any(self.nid_is_on_gpu[(self.device_index + 1) % 2::2]))

    def sync_cache_read_end(self, index: int):
        """Releease relevant synchronization resources related to self.sync_cache_read_start

        Args:
            index (int): Device index to be read from
        """
        # with Timer('read lock unlock'):
            # self.cache_manager.thread_exit()
        if self.use_locking:
            # [torch.cuda.synchronize(i) for i in range(torch.cuda.device_count())]
            self.cache_manager.read_unlock(index)
        else:
            self.cache_manager.thread_exit(index, self.device_index, self.executor_id)
        
class FrequencySynchronousCache(CountingFeatServer):
    cache_name = 'freq_sync'
    # TODO tidy this up, no need to num total nodes again here
    def init_counts(self, num_total_nodes):
        super().init_counts(num_total_nodes)
        self.is_cache_candidate = self.caches[self.store_id].is_cache_candidate
        self.topk_stream = None
        self.update_stream = None    

    def update_cache(self):
        if not self.is_leader:
            return

        cache_size = self.caches[self.store_id].cache_size

        if self.topk_stream is None:
            self.topk_stream = torch.cuda.Stream(device=self.device)
        if self.update_stream is None:
            self.update_stream = torch.cuda.Stream(device=self.device)
    
        torch.zeros(self.num_nodes, out=self.is_cache_candidate, dtype=torch.bool, device=self.device)

        if len(self.caches) > 1:
            # total_counts = functools.reduce(torch.add, [peer.counts.to(self.device) for peer in self.caches])
            # big_graph_arange = torch.arange(self.num_nodes, device=self.device)
            # part_nids = big_graph_arange[big_graph_arange % len(self.caches) == self.device_index]
            # _, most_common_idxs = torch.topk(total_counts[part_nids], self.cache_size, sorted=False)
            # most_common_nids = part_nids[most_common_idxs]
            part_counts = self.counts[self.device_index::len(self.caches)]
            _, top_part_idxs = torch.topk(part_counts, cache_size, sorted=False)
            self.most_common_nids = top_part_idxs * len(self.caches) + self.device_index
        else:
            _, self.most_common_nids = torch.topk(self.counts.to(self.device, non_blocking=True), cache_size, sorted=False)

        self.is_cache_candidate[self.most_common_nids] = True
        self.topk_processed = True
        torch.div(self.counts, 2, rounding_mode='floor', out=self.counts)

    def get_features(self, node_ids: torch.LongTensor, feats: List[str], mfgs: Optional[dgl.DGLGraph]=None):
        """Get features for a list of nodes.

        Features are fetched from GPU memory if cached, otherwise from CPU memory.

        Args:
            node_ids (torch.Tensor): A 1-D tensor of node IDs.
            feats (List[str]): List of strings corresponding to feature keys that should be fetched.
        """
        gpu_nids = node_ids
        with Timer('update counts'):
            if self.is_leader:
                self.counts[gpu_nids] += 1

        res_tensor_dict, mfgs, newly_transferred_nids, newly_transferred_feats = self._get_features(node_ids, feats, mfgs)

        with Timer('cache update', track_cuda=True):
            torch.cuda.synchronize()
            cache = self.caches[self.store_id].cache
            cache_mask = self.caches[self.store_id].cache_mask
            cache_mapping = self.caches[self.store_id].cache_mapping
            cache_size = self.caches[self.store_id].cache_size

            new_feats = newly_transferred_feats
            new_nids = newly_transferred_nids
            new_nid_mask = self.is_cache_candidate[new_nids]

            nids_to_add = new_nids[new_nid_mask]
            new_feats = new_feats[new_nid_mask]

            replace_nid_mask = cache_mask & ~self.is_cache_candidate

            replace_nids = replace_nid_mask.nonzero()
            replace_nids = replace_nids.reshape(replace_nids.shape[0])

                # # with Timer('truncate'):
            num_to_add = min(replace_nids.shape[0], nids_to_add.shape[0], cache_size)
            replace_nids = replace_nids[:num_to_add]
            nids_to_add = nids_to_add[:num_to_add]

            cache_mask[replace_nids] = False
            with Timer('meta update'):
                cache_mask[replace_nids] = False
                cache_mask[nids_to_add] = True
                # self.nid_is_on_gpu.copy_(cache_mask_device, non_blocking=True)
            cache_slots = cache_mapping[replace_nids]

            cache_mapping[replace_nids] = -1
            cache_mapping[nids_to_add] = cache_slots

            for feat in cache.keys():
                old_shape = cache[feat].shape
                    # # with Timer('actual move'):
                    # # Recall the above truncation - the features we want will be at the front of the result tensor
                cache[feat][cache_slots] = newly_transferred_feats[:num_to_add]
                assert(cache[feat].shape == old_shape)

            cache_mask[nids_to_add] = True

        return res_tensor_dict, mfgs
