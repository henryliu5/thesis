import torch
import dgl
from typing import List, Optional, Dict
from fast_inference.timer import Timer, export_dict_as_pd
from fast_inference_cpp import CacheManager
import time

class FeatureServer:
    def __init__(self, 
                 num_nodes: int,
                 features: Dict[str, torch.Tensor], 
                 device: torch.device or str,
                 track_features: List[str],
                 use_pinned_mem: bool = True,
                 profile_hit_rate: bool = False,
                 pinned_buf_size: int = 150_000,
                 peer_lock = None):
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
        self.nid_is_on_gpu = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        # TODO should this go on GPU?
        self.cache_mapping = - \
            torch.ones(num_nodes, device=self.device).long()
        self.cache = {}
        self.use_pinned_mem = use_pinned_mem
        self.profile = profile_hit_rate
        self.profile_info = {'request_size': [], 'cache_hits': [], 'hit_rate': []}

        # Pinned memory buffers for placing gathered CPU features prior to CPU->GPU copy
        self.pinned_buf_dict = {}
        # NOTE allocate "small" pinned buffers to place features that will be transferred
        for feature in track_features:
            # TODO make these buffers work with features that are not 1D (see pytest test)
            self.pinned_buf_dict[feature] = torch.empty((pinned_buf_size, features[feature].shape[1]), dtype=torch.float, pin_memory=True)

        self.peers = None
        self.peer_streams = None
        self.peer_lock = peer_lock
        self.lock_conflicts = 0

    def set_peer_group(self, peers):
        self.peers = peers
        print('Setting peers for FeatureStore', self.device_index, self.peers)

    def get_peer_features(self, node_ids: torch.LongTensor, feat: str):
        if self.peer_streams is None:
            self.peer_streams = [torch.cuda.Stream(device=peer.device) for peer in self.peers]

        assert(node_ids.device == self.device)
        if self.peers is None:
            return
        
        result_masks = []
        result_features = []
        # Check which nodes are on GPUs
        gpu_nids = node_ids[self.nid_is_on_gpu[node_ids]]

        dur = 0 
        num_peers = len(self.peers)
        for i in range(num_peers):
            s = time.perf_counter()
            self.peer_lock[i].acquire()
            dur += time.perf_counter() - s

            peer = self.peers[i]
            # Only transfer node ids that belong to that GPU
            peer_mask = gpu_nids % num_peers == i
            # peer_mask = torch.ones(gpu_nids.shape, dtype=torch.bool, device=self.device)
            peer_nids = gpu_nids[peer_mask].to(peer.device)
            torch.cuda.current_stream().synchronize() # Must explicitly wait for nids to reach peer

            with torch.cuda.stream(self.peer_streams[i]):
                mapping = peer.cache_mapping[peer_nids]
                # assert(torch.all(mapping >= 0))
                result_features.append(peer.cache[feat][mapping].to(self.device))

            result_masks.append(peer_mask)

            self.peer_lock[i].release()

        [stream.synchronize() for stream in self.peer_streams]

        if dur > 0.0005:
            print('Waited for lock', dur)
            self.lock_conflicts += 1

        return result_masks, result_features

    def get_features(self, node_ids: torch.LongTensor, feats: List[str], mfgs: Optional[dgl.DGLGraph]=None):
        """Get features for a list of nodes.

        Features are fetched from GPU memory if cached, otherwise from CPU memory.

        Args:
            node_ids (torch.Tensor): A 1-D tensor of node IDs.
            feats (List[str]): List of strings corresponding to feature keys that should be fetched.
        """
        assert(node_ids.device != torch.device('cpu'))
        if mfgs is None:
            mfgs = []

        with Timer('get_features()'):
            res = {}

            with Timer('get peer features'):
                self.get_peer_features(node_ids, feats[0])

            # Used to mask this particular request - not to mask the cache!!
            with Timer('compute gpu/cpu mask'):
                gpu_mask = self.nid_is_on_gpu[node_ids]
                cpu_mask = ~gpu_mask

            if self.profile:
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
                    m = node_ids[cpu_mask].cpu()
                    # TODO add parameter to control "use_pinned_mem"
                    # Perform resizing if necessary
                    if self.use_pinned_mem:
                        if m.shape[0] > self.pinned_buf_dict[feat].shape[0]:
                            self.pinned_buf_dict[feat] = self.pinned_buf_dict[feat].resize_((m.shape[0], self.pinned_buf_dict[feat].shape[1]))
                        required_cpu_features = self.pinned_buf_dict[feat].narrow(0, 0, m.shape[0])

                with Timer('feature gather'):
                    if self.use_pinned_mem:
                        # Places indices directly into pinned memory buffer
                        torch.index_select(self.features[feat], 0, m, out=required_cpu_features)
                    else:
                        #"slow mode"
                        required_cpu_features = torch.index_select(self.features[feat], 0, m)

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
                        mapping = self.cache_mapping[node_ids[gpu_mask]]
                        assert(torch.all(mapping >= 0))
                        required_gpu_features = self.cache[feat][mapping]
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
        # Reset all
        self.nid_is_on_gpu[:] = False
        self.nid_is_on_gpu[node_ids] = True
        self.cache_mapping[node_ids] = torch.arange(
            self.cache_size, device=self.device)

        for feat in feats:
            self.cache[feat] = self.features[feat][node_ids].to(self.device)

    def export_profile(self, path, current_config):
        if self.profile:
            export_dict_as_pd(self.profile_info, path, current_config)
        else:
            print('FeatureServer.export_profile called but profiling disabled!')

    def init_counts(self, *args):
        pass

    def update_cache(self, *args):
        pass

    def compute_topk(self, *args):
        pass

class CountingFeatServer(FeatureServer):
    # TODO tidy this up, no need to num total nodes again here
    def init_counts(self, num_total_nodes):
        self.num_total_nodes = num_total_nodes
        self.counts = torch.zeros(num_total_nodes, dtype=torch.long, device=self.device)

        self.topk_stream = torch.cuda.Stream(device=self.device)
        self.update_stream = torch.cuda.Stream(device=self.device)

        self.most_common_nids = None
        self.topk_started = False

    def compute_topk(self):
        # with torch.cuda.stream(self.topk_stream):
        #     _, self.most_common_nids = torch.topk(self.counts.to(self.device, non_blocking=True), self.cache_size, sorted=False)
        #     self.topk_started = True
        pass

    def update_cache(self, feats):
        with torch.cuda.stream(self.update_stream):
            v, most_common_nids = torch.topk(self.counts.to(self.device), self.cache_size, sorted=False)
            assert(torch.all(self.counts >= 0))
            # # Resets cache mask (nothing stored anymore)
            # if not self.topk_started:
            #     _, most_common_nids = torch.topk(self.counts.to(self.device), self.cache_size, sorted=False)
            # else:
            #     torch.cuda.current_stream().wait_stream(self.topk_stream)
            #     most_common_nids = self.most_common_nids

            # cache_mask_device = self.nid_is_on_gpu.to(self.device, non_blocking=True)
            cache_mask_device = self.nid_is_on_gpu
            most_common_mask = torch.zeros(self.num_total_nodes, dtype=torch.bool, device=self.device)
            most_common_mask[most_common_nids] = True

            # Mask for node ids that need features to be transferred
            # (new entrants to cache)            
            requires_update_mask = torch.logical_and(most_common_mask, torch.logical_not(cache_mask_device))

            # Indices of who can be replaced in the cache
            replace_nids_mask = torch.logical_and(~most_common_mask, cache_mask_device)
            requires_update_cache_idx = self.cache_mapping[replace_nids_mask]

            for feat in feats:
                old_shape = self.cache[feat].shape
                self.cache[feat][requires_update_cache_idx] = self.features[feat][requires_update_mask.cpu()].to(self.device)
                assert(self.cache[feat].shape == old_shape)

            self.cache_mapping[requires_update_mask] = requires_update_cache_idx
            #!! Need this weird copy_ to perform device to host non blocking transfer (and into pinned memory buffer)
            self.nid_is_on_gpu = most_common_mask
            self.cache_mapping[~most_common_mask] = -1
            torch.div(self.counts, 2, rounding_mode='floor', out=self.counts)
            
        torch.cuda.current_stream().wait_stream(self.update_stream)

    def get_features(self, node_ids: torch.LongTensor, feats: List[str], mfgs: Optional[dgl.DGLGraph]=None):
        """Get features for a list of nodes.

        Features are fetched from GPU memory if cached, otherwise from CPU memory.

        Args:
            node_ids (torch.Tensor): A 1-D tensor of node IDs.
            feats (List[str]): List of strings corresponding to feature keys that should be fetched.
        """
        with Timer('update counts'):
            self.counts[node_ids] += 1

        return super().get_features(node_ids, feats, mfgs)
    
class LFUServer(FeatureServer):

    def init_counts(self, num_total_nodes):
        self.num_total_nodes = num_total_nodes
        self.counts = torch.zeros(num_total_nodes, device=self.device)

    def update_cache(self, *args):
        torch.div(self.counts, 2, rounding_mode='floor', out=self.counts)

    def get_features(self, node_ids: torch.LongTensor, feats: List[str], mfgs: Optional[dgl.DGLGraph]=None):
        """Get features for a list of nodes.

        Features are fetched from GPU memory if cached, otherwise from CPU memory.

        Args:
            node_ids (torch.Tensor): A 1-D tensor of node IDs.
            feats (List[str]): List of strings corresponding to feature keys that should be fetched.
        """
        cpu_node_ids = node_ids.cpu()
        with Timer('update counts'):
            self.counts[node_ids] += 1

        with Timer('super get features'):
            res_dict, res_mfg = super().get_features(node_ids, feats, mfgs)

        with Timer('LFU update'):
            # Perform LFU update
            # Admission policy is simply allow everything in
            gpu_mask = self.nid_is_on_gpu[node_ids]
            cpu_mask = ~gpu_mask

            # Will want to add cache misses
            nids_to_add = node_ids[cpu_mask]

            for feat in feats:
                cache_size = self.cache[feat].shape[0]
                if nids_to_add.shape[0] > cache_size:
                    # Truncate if necessary, just take whatever first firts
                    nids_to_add = nids_to_add[:cache_size]

                count_of_cache_residents = self.counts[self.nid_is_on_gpu]
                resident_mapping = torch.arange(self.nid_is_on_gpu.shape[0], device=self.device)[self.nid_is_on_gpu]
                # Replace lowest count
                _, replace_residents = torch.topk(count_of_cache_residents, k=nids_to_add.shape[0], largest=False, sorted=True)
                replace_nids = resident_mapping[replace_residents]

                cache_slots = self.cache_mapping[replace_nids]
                self.nid_is_on_gpu[replace_nids] = False
                self.cache_mapping[replace_nids] = -1

                self.nid_is_on_gpu[nids_to_add] = True
                self.cache_mapping[nids_to_add] = cache_slots

                old_shape = self.cache[feat].shape
                
                # Recall the above truncation - the features we want will be at the front of the result tensor
                self.cache[feat][cache_slots] = res_dict[feat][cpu_mask][:cache_size]
                assert(self.cache[feat].shape == old_shape)

        return res_dict, res_mfg


class ManagedCacheServer(FeatureServer):

    def init_counts(self, num_total_nodes):
        self.num_total_nodes = num_total_nodes
        self.counts = torch.zeros(num_total_nodes, dtype=torch.short, device=self.device)

        self.topk_stream = torch.cuda.Stream(device=self.device)
        self.update_stream = torch.cuda.Stream(device=self.device)

        self.is_cache_candidate = None
        self.most_common_nids = None
        self.topk_started = False
        self.topk_processed = False

        self.big_graph_arange = torch.arange(num_total_nodes, device=self.device)


    def _start_manager(self):
        self.num_total_nodes = self.num_nodes
        self.cache_manager = CacheManager(self.num_total_nodes, self.cache_size, -1, -1, 0, True)


    def set_static_cache(self, node_ids: torch.Tensor, feats: List[str]):
        """Define a static cache using the given node ids.

        Args:
            node_ids (torch.Tensor): Elements should be node ids whose features are to be cached in GPU memory.
            feats (List[str]): List of strings corresponding to feature keys that should be cached.
        """
        self.cache_size = node_ids.shape[0]
        # Reset all
        self.nid_is_on_gpu[:] = False
        self.nid_is_on_gpu[node_ids] = True
        self.cache_mapping[node_ids] = torch.arange(
            self.cache_size, device=self.device)

        self.reverse_mapping = node_ids
        for feat in feats:
            self.cache[feat] = self.features[feat][node_ids].to(self.device)

            # TODO support more than 1 featuer type
            break

    def start_manager(self):
        for feat in self.features:
            self._start_manager()
            self.cache_manager.set_cache(self.features[feat], self.nid_is_on_gpu, 
                                        self.cache_mapping, self.reverse_mapping.to(self.device), self.cache[feat])
            break

    def compute_topk(self):
        with torch.cuda.stream(self.topk_stream):
            if self.is_cache_candidate is None:
                self.is_cache_candidate = torch.zeros(self.num_total_nodes, dtype=torch.bool, device=self.device)
                self.cache_manager.set_cache_candidates(self.is_cache_candidate)
            else:
                torch.zeros(self.num_total_nodes, out=self.is_cache_candidate, dtype=torch.bool, device=self.device)

            _, self.most_common_nids = torch.topk(self.counts.to(self.device, non_blocking=True), self.cache_size, sorted=False)

            self.topk_started = True
            self.topk_processed = False

    def update_cache(self, feats):
        # if self.topk_started:
        #     # TODO figure out why this doesn't work when put by placing features in the queue
        #     if not self.topk_processed:
        #         torch.cuda.current_stream().wait_stream(self.topk_stream)
        #         #!! This first line is kinda weird but goes here to allow
        #         #!! self.most_common_nids to be computed async in self.topk_stream
        #         self.is_cache_candidate[self.most_common_nids] = True
        #         self.topk_processed = True
        pass
        

    def get_features(self, node_ids: torch.LongTensor, feats: List[str], mfgs: Optional[dgl.DGLGraph]=None):
        """Get features for a list of nodes.

        Features are fetched from GPU memory if cached, otherwise from CPU memory.

        Args:
            node_ids (torch.Tensor): A 1-D tensor of node IDs.
            feats (List[str]): List of strings corresponding to feature keys that should be fetched.
        """
        gpu_nids = node_ids
        with Timer('update counts'):
            self.counts[gpu_nids] += 1

        if mfgs is None:
            mfgs = []

        with Timer('get_features()'):
            res = {}

            self.cache_manager.thread_enter()
            # self.cache_manager.lock()

            # Used to mask this particular request - not to mask the cache!!
            with Timer('compute gpu/cpu mask'):
                gpu_mask = self.nid_is_on_gpu[node_ids]
                cpu_mask = ~gpu_mask

            if self.profile:
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
                    # TODO add parameter to control "use_pinned_mem"
                    # Perform resizing if necessary
                    if self.use_pinned_mem:
                        if m.shape[0] > self.pinned_buf_dict[feat].shape[0]:
                            self.pinned_buf_dict[feat] = self.pinned_buf_dict[feat].resize_((m.shape[0], self.pinned_buf_dict[feat].shape[1]))
                        required_cpu_features = self.pinned_buf_dict[feat].narrow(0, 0, m.shape[0])

                with Timer('feature gather'):
                    if self.use_pinned_mem:
                        # Places indices directly into pinned memory buffer
                        torch.index_select(self.features[feat], 0, m.cpu(), out=required_cpu_features)
                    else:
                        #"slow mode"
                        required_cpu_features = torch.index_select(self.features[feat], 0, m.cpu())

                with Timer('CPU-GPU copy', track_cuda=True):
                    # Copy CPU features
                    cpu_feats = required_cpu_features.to(
                        self.device, non_blocking=True)
                    res_tensor[cpu_mask] = cpu_feats
                    # Copy MFGs
                    mfgs = [mfg.to(self.device) for mfg in mfgs]

                with Timer('move cached features', track_cuda=True):
                    # Features from GPU mem
                    # self.cache_mapping maps the global node id to the respective index in the cache
                    if feat in self.cache: # hacky drop in for torch.any(gpu_mask)
                        mapping = self.cache_mapping[node_ids[gpu_mask]]
                        assert(torch.all(mapping >= 0))
                        required_gpu_features = self.cache[feat][mapping]
                        res_tensor[gpu_mask] = required_gpu_features

                self.cache_manager.thread_exit()
                # self.cache_manager.unlock()
                res[feat] = res_tensor

                with Timer('cache update'):
                    if self.topk_started:
                        if not self.topk_processed:
                            torch.cuda.current_stream().wait_stream(self.topk_stream)
                            with torch.cuda.stream(self.update_stream):
                                #!! This first line is kinda weird but goes here to allow
                                #!! self.most_common_nids to be computed async in self.topk_stream
                                self.is_cache_candidate[self.most_common_nids] = True
                            self.topk_processed = True
                            torch.div(self.counts, 2, rounding_mode='floor', out=self.counts)
                            torch.cuda.current_stream().wait_stream(self.update_stream)

                        # with Timer('place in queue'):
                        # !! WARNING: Must use "m" here!! Since the node ids and mask are on GPU, the CPU node id tensor
                        # !! must be fully materialized by the time the tensor is placed on the queue

                        self.cache_manager.place_feats_in_queue(cpu_feats, m)

                        # # Can comment in below and comment out above to use syncrhonous update (async in stream)
                        # with torch.cuda.stream(self.update_stream):
                        #     new_feats = cpu_feats
                        #     new_nids = m
                        #     new_nid_mask = self.is_cache_candidate[new_nids]

                        #     nids_to_add = new_nids[new_nid_mask]
                        #     new_feats = new_feats[new_nid_mask]

                        #     replace_nid_mask = self.nid_is_on_gpu & ~ self.is_cache_candidate

                        #     replace_nids = replace_nid_mask.nonzero()
                        #     replace_nids = replace_nids.reshape(replace_nids.shape[0])


                        #         # # with Timer('truncate'):
                        #     num_to_add = min(replace_nids.shape[0], nids_to_add.shape[0], self.cache_size)
                        #     replace_nids = replace_nids[:num_to_add]
                        #     nids_to_add = nids_to_add[:num_to_add]

                        #     self.nid_is_on_gpu[replace_nids] = False
                        #         # # with Timer('meta update'):
                        #         # cache_mask_device[replace_nids] = False
                        #         # cache_mask_device[nids_to_add] = True
                        #         # self.nid_is_on_gpu.copy_(cache_mask_device, non_blocking=True)
                        #     cache_slots = self.cache_mapping[replace_nids]

                        #     self.cache_mapping[replace_nids] = -1
                        #     self.cache_mapping[nids_to_add] = cache_slots

                        #     old_shape = self.cache[feat].shape
                        #         # # with Timer('actual move'):
                        #         # # Recall the above truncation - the features we want will be at the front of the result tensor
                        #     self.cache[feat][cache_slots] = res[feat][cpu_mask][:num_to_add]
                        #     assert(self.cache[feat].shape == old_shape)

                        #     self.nid_is_on_gpu[nids_to_add] = True

        return res, mfgs
