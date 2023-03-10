import torch
import dgl
from typing import List, Optional
from fast_inference.timer import Timer, export_dict_as_pd

class FeatureServer:
    def __init__(self, g: dgl.DGLGraph, 
                 device: torch.device or str,
                 track_features: List[str],
                 use_pinned_mem: bool = True,
                 profile_hit_rate: bool = False,
                 pinned_buf_size: int = 150_000):
        """ Initializes a new FeatureServer

        Args:
            g (dgl.DGLGraph): Graph whose features are to be served. Graph should be on CPU.
            device (torch.device): Device where feature server should store cache
        """
        assert (g.device == torch.device('cpu'))
        self.g = g
        self.device = device
        self.nid_is_on_gpu = torch.zeros(g.num_nodes()).bool()
        # TODO should this go on GPU?
        self.cache_mapping = - \
            torch.ones(g.num_nodes(), device=self.device).long()
        self.cache = {}
        self.use_pinned_mem = use_pinned_mem
        self.profile = profile_hit_rate
        self.profile_info = {'request_size': [], 'cache_hits': [], 'hit_rate': []}

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
                self.profile_info['request_size'].append(node_ids.shape[0])
                cache_hits = gpu_mask.int().sum().item()
                self.profile_info['cache_hits'].append(cache_hits)
                self.profile_info['hit_rate'].append(cache_hits / node_ids.shape[0])

            for feat in feats:
                feat_shape = list(self.g.ndata[feat].shape[1:])
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
                        torch.index_select(self.g.ndata[feat], 0, m, out=required_cpu_features)
                    else:
                        #"slow mode"
                        required_cpu_features = torch.index_select(self.g.ndata[feat], 0, m)

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
            self.cache[feat] = self.g.ndata[feat][node_ids].to(self.device)

    def export_profile(self, path, current_config):
        if self.profile:
            export_dict_as_pd(self.profile_info, path, current_config)
        else:
            print('FeatureServer.export_profile called but profiling disabled!')

    def init_counts(self, *args):
        pass

    def update_cache(self, *args):
        pass

class CountingFeatServer(FeatureServer):

    def init_counts(self, num_total_nodes):
        self.num_total_nodes = num_total_nodes
        self.counts = torch.zeros(num_total_nodes)

    def update_cache(self, feats):
        # Resets cache mask (nothing stored anymore)
        _, most_common_nids = torch.topk(self.counts, self.cache_size, sorted=False)
        # # Updates to most common in based on self.counts
        # self.set_static_cache(most_common_nids, ['feat'])

        most_common_mask = torch.zeros(self.g.num_nodes(), dtype=torch.bool)
        most_common_mask[most_common_nids] = True
        # Mask for node ids that need features to be transferred
        # (new entrants to cache)
        requires_update_mask = torch.logical_and(most_common_mask, torch.logical_not(self.nid_is_on_gpu))

        # Indices of who can be replaced in the cache
        replace_nids_mask = torch.logical_and(~most_common_mask, self.nid_is_on_gpu)
        requires_update_cache_idx = self.cache_mapping[replace_nids_mask]

        for feat in feats:
            old_shape = self.cache[feat].shape
            self.cache[feat][requires_update_cache_idx] = self.g.ndata[feat][requires_update_mask].to(self.device)
            assert(self.cache[feat].shape == old_shape)

        self.nid_is_on_gpu[most_common_mask] = True
        self.cache_mapping[requires_update_mask] = requires_update_cache_idx
        self.nid_is_on_gpu[~most_common_mask] = False
        self.cache_mapping[~most_common_mask] = -1
        # self.counts *= 0
        torch.div(self.counts, 2, rounding_mode='floor', out=self.counts)

    def get_features(self, node_ids: torch.LongTensor, feats: List[str], mfgs: Optional[dgl.DGLGraph]=None):
        """Get features for a list of nodes.

        Features are fetched from GPU memory if cached, otherwise from CPU memory.

        Args:
            node_ids (torch.Tensor): A 1-D tensor of node IDs.
            feats (List[str]): List of strings corresponding to feature keys that should be fetched.
        """
        node_ids = node_ids.cpu()
        with Timer('update counts'):
            self.counts[node_ids] += 1
        return super().get_features(node_ids, feats, mfgs)
    
class LFUServer(FeatureServer):

    def init_counts(self, num_total_nodes):
        self.num_total_nodes = num_total_nodes
        self.counts = torch.zeros(num_total_nodes)

    def update_cache(self, *args):
        torch.div(self.counts, 2, rounding_mode='floor', out=self.counts)

    def get_features(self, node_ids: torch.LongTensor, feats: List[str], mfgs: Optional[dgl.DGLGraph]=None):
        """Get features for a list of nodes.

        Features are fetched from GPU memory if cached, otherwise from CPU memory.

        Args:
            node_ids (torch.Tensor): A 1-D tensor of node IDs.
            feats (List[str]): List of strings corresponding to feature keys that should be fetched.
        """
        node_ids = node_ids.cpu()
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
                resident_mapping = torch.arange(self.nid_is_on_gpu.shape[0])[self.nid_is_on_gpu]
                # Replace lowest count
                _, replace_residents = torch.topk(count_of_cache_residents, k=nids_to_add.shape[0], largest=False, sorted=False)
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
    
import ray

@ray.remote
class FeatureCounter:
    def __init__(self, num_total_nodes, cache_size):
        self.num_total_nodes = num_total_nodes
        self.counts = torch.zeros(num_total_nodes)
        self.cache_size = cache_size
        self.is_cache_candidate = None
        self.counts_still_zero = True
    
    def incr_counts(self, node_ids):
        self.counts_still_zero = False
        self.counts[torch.from_numpy(node_ids)] += 1
    
    def window_update(self):
        torch.div(self.counts, 2, rounding_mode='floor', out=self.counts)
    
    def determine_cache_candidates(self):
        _, cache_candidates = torch.topk(self.counts, self.cache_size, sorted=False)
        self.is_cache_candidate = torch.zeros(self.counts.shape[0], dtype=bool)
        self.is_cache_candidate[cache_candidates] = True

    def get_cache_candidates(self):
        if self.counts_still_zero:
            return None
        return self.is_cache_candidate.numpy(force=False)
    

class HybridServer(FeatureServer):
    ''' Policy that performs admission/evict per request, but makes admission/eviction
        decision based on frequency statistic from past k requests.
    '''

    def init_counts(self, num_total_nodes):
        self.num_total_nodes = num_total_nodes
        self.counts = torch.zeros(num_total_nodes)
        self.big_graph_arange = torch.arange(num_total_nodes)
        self.remote_counter = FeatureCounter.remote(num_total_nodes, self.cache_size)
        self.remote_update_future = None
        
        self.copy_stream = torch.cuda.Stream(device='cuda')
        self.copy_stream2 = torch.cuda.Stream(device='cuda')

    def update_cache(self, *args):
        self.remote_counter.window_update.remote()
        # Start new update
        self.remote_counter.determine_cache_candidates.remote()
        if self.remote_update_future:
            self.is_cache_candidate = torch.from_numpy(ray.get(self.remote_update_future))
            self.remote_update_future = None

        self.remote_update_future = self.remote_counter.get_cache_candidates.remote()

        # torch.div(self.counts, 2, rounding_mode='floor', out=self.counts)
        # _, cache_candidates = torch.topk(self.counts, self.cache_size, sorted=False)
        # self.is_cache_candidate = torch.zeros(self.counts.shape[0], dtype=bool)
        # self.is_cache_candidate[cache_candidates] = True

    def get_features(self, node_ids: torch.LongTensor, feats: List[str], mfgs: Optional[dgl.DGLGraph]=None):
        """Get features for a list of nodes.

        Features are fetched from GPU memory if cached, otherwise from CPU memory.

        Args:
            node_ids (torch.Tensor): A 1-D tensor of node IDs.
            feats (List[str]): List of strings corresponding to feature keys that should be fetched.
        """
        gpu_node_ids = node_ids.to(self.device)
        node_ids = node_ids.cpu()
        with Timer('update counts'):
            self.remote_counter.incr_counts.remote(node_ids.numpy(force = False))
            # self.counts[node_ids] += 1

        if mfgs is None:
            mfgs = []

        with Timer('get_features()'):
            res = {}
            copied_features = {}

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
                feat_shape = list(self.g.ndata[feat].shape[1:])
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
                    if self.use_pinned_mem:
                        # Places indices directly into pinned memory buffer
                        torch.index_select(self.g.ndata[feat], 0, m, out=required_cpu_features)
                    else:
                        #"slow mode"
                        required_cpu_features = torch.index_select(self.g.ndata[feat], 0, m)

                with Timer('CPU-GPU copy', track_cuda=True):
                    # Need to get small copies out of the way so computations can be done async
                    # with big transfer from feature copy
                    with Timer('get gpu mask cache indices'):
                        gpu_ids = gpu_node_ids[gpu_mask]
                        gpu_mask = gpu_mask.to(self.device)
                        moved_cpu_mask = cpu_mask.to(self.device)

                    with Timer('mfg copy'):
                        if mfgs[0].device == torch.device('cpu'):
                            # Copy MFGs
                            mfgs = [mfg.to(self.device) for mfg in mfgs]

                    with torch.cuda.stream(self.copy_stream):
                        with Timer('actual copy'):
                            # Copy CPU features
                            cpu_copied_features = required_cpu_features.to(
                                self.device, non_blocking=True)

                with Timer('move cached features', track_cuda=True):
                    # Features from GPU mem
                    # self.cache_mapping maps the global node id to the respective index in the cache
                    if feat in self.cache: # hacky drop in for torch.any(gpu_mask)
                        with torch.cuda.stream(self.copy_stream2):
                            with Timer('get gpu cache mapping'):
                                gpu_mapping = self.cache_mapping[gpu_ids]
                            assert(torch.all(gpu_mapping >= 0))
                            required_gpu_features = self.cache[feat][gpu_mapping]
                            res_tensor[gpu_mask] = required_gpu_features
            
                # with Timer('CPU-GPU copy finish', track_cuda=True):
                #     with Timer('index into res tensor'):
                #         # torch.cuda.current_stream().wait_stream(self.copy_stream)
                #         # torch.cuda.current_stream().wait_stream(self.copy_stream2)
                #         res_tensor[moved_cpu_mask] = cpu_copied_features
    
                res[feat] = res_tensor
                copied_features[feat] = (cpu_copied_features, moved_cpu_mask)
                # copied_features[feat] = (cpu_copied_features, cpu_mask, gpu_mapping, gpu_mask)

        with Timer('hybrid cache update'):
            if hasattr(self, 'is_cache_candidate'):
                cpu_mask = cpu_mask & self.is_cache_candidate[node_ids]

                # Will want to add cache misses
                nids_to_add = node_ids[cpu_mask]

                for feat in feats:
                    cache_size = self.cache[feat].shape[0]
                    if nids_to_add.shape[0] > cache_size:
                        # Truncate if necessary, just take whatever first
                        nids_to_add = nids_to_add[:cache_size]

                    # nids that are in cache and can be replaced in this epoch
                    replace_nid_mask = self.nid_is_on_gpu & ~self.is_cache_candidate
                                                # total # of nodes    
                    replace_nids = self.big_graph_arange[replace_nid_mask][:nids_to_add.shape[0]]

                    self.nid_is_on_gpu[replace_nids] = False
                    self.nid_is_on_gpu[nids_to_add] = True
                    #! NOTE: Synchronization point since cache mapping is on GPU
                    cache_slots = self.cache_mapping[replace_nids]
                    self.cache_mapping[replace_nids] = -1
                    self.cache_mapping[nids_to_add] = cache_slots

                    with Timer('finish copy cpu-gpu'):
                        torch.cuda.current_stream().wait_stream(self.copy_stream)
                        cpu_copied_features, orig_cpu_mask = copied_features[feat]
                        res_tensor[orig_cpu_mask] = cpu_copied_features

                    old_shape = self.cache[feat].shape
                    # Recall the above truncation - the features we want will be at the front of the result tensor
                    self.cache[feat][cache_slots] = res[feat][cpu_mask][:cache_size]
                    assert(self.cache[feat].shape == old_shape)
            else:
                with Timer('finish copy cpu-gpu'):
                    torch.cuda.current_stream().wait_stream(self.copy_stream)
                    cpu_copied_features, orig_cpu_mask = copied_features[feat]
                    res_tensor[orig_cpu_mask] = cpu_copied_features
        #         with Timer('required feat index'):
        #             required_gpu_features = self.cache[feat][gpu_mapping]
        #         with Timer('move into result tensor'):
        #             res_tensor[gpu_mask] = required_gpu_features

        return res, mfgs