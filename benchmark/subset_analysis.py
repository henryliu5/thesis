''' Utiliy to check the hit rate of static cache'''
from fast_inference.dataset import InferenceDataset
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import math
from tqdm import tqdm
import dgl
import torch
import math

# def get_slice(tensor, partition, total_partitions, cache_size):

# class CountingCache:
#     ''' LFU-like cache, can be updated with update_cache '''
#     def __init__(self, init_indices, num_total_nodes):
#         self.num_total_nodes = num_total_nodes
#         self.cache_size = init_indices.shape[0]
#         self.cache_mask = torch.zeros(num_total_nodes, dtype=torch.bool)
#         self.counts = torch.zeros(num_total_nodes)

#         self.cache_mask[init_indices] = True

#     def update_cache(self):
#         # Resets cache mask (nothing stored anymore)
#         self.cache_mask = torch.zeros(self.num_total_nodes, dtype=torch.bool)
#         _, most_common_nids = torch.topk(self.counts, self.cache_size, sorted=False)
#         # Updates to most common in based on self.counts
#         self.cache_mask[most_common_nids] = True
#         self.counts *= 0
    
#     def check_cache(self, request_nids) -> int:
#         ''' Return how many of request_nids are in the cache (how many cache hits).
            
#             Also updates the summary statistics.
#         '''
#         self.counts[request_nids] += 1
#         in_mask = self.cache_mask[request_nids]
#         return torch.count_nonzero(in_mask.int())

class PartitionCache:
    UPDATES_TO_DECREMENT = 5
    ''' LFU-like cache, can be updated with update_cache '''
    # NOTE: very good cache hit rate with update every 250, 5 partitions``
    def __init__(self, init_indices, num_total_nodes, partitions = 32):
        self.num_total_nodes = num_total_nodes
        self.cache_size = init_indices.shape[0]
        self.cache_mask = torch.zeros(num_total_nodes, dtype=torch.bool)
        self.counts = torch.zeros(num_total_nodes)

        # self.partition_size =  math.ceil(self.cache_size / partitions)
        self.partition_size = self.cache_size // partitions
        # First partition will be larger than rest
        self.partition_offset = self.cache_size % partitions
        self.partition_mapping = torch.repeat_interleave(torch.arange(partitions, dtype=torch.long), self.partition_size)
        self.partition_mapping = torch.cat((torch.zeros(self.partition_offset, dtype=torch.long), self.partition_mapping))


        self.cache_mask[init_indices] = True
        self.cache_mapping = -1 * torch.ones(num_total_nodes).long()
        self.cache_mapping[init_indices] = torch.arange(self.cache_size)

        self.reverse_mapping = init_indices

        self.hits_per_partition = torch.zeros(partitions)
        self.partitions = partitions
        
        self.num_updates = 0

    def update_cache(self):
        # Compute worst performing cache partition
        _, worst_partition = torch.min(self.hits_per_partition, dim=0)
        # worst_partition * self.partition_size : min((worst_partition + 1) * self.partition_size)

        if worst_partition != 0:
            cache_start_idx = worst_partition * self.partition_size + self.partition_offset
            cache_stop_idx = cache_start_idx + self.partition_size
        else:
            cache_start_idx = 0
            cache_stop_idx = self.partition_size + self.partition_offset

        remove_nids = self.reverse_mapping[cache_start_idx : cache_stop_idx].clone()

        values, most_common_nids = torch.topk(self.counts, self.cache_size, sorted=True)

        # Make it okay to keep the best parts of the worst partition
        self.cache_mask[remove_nids] = False

        # # Updates to most common in based on self.counts
        best_not_in_cache = most_common_nids[self.cache_mask[most_common_nids] == False][:self.partition_size]
        # print('optimal cache frequencies', values)

        # v, _ = torch.sort(self.counts[self.reverse_mapping], descending=True)
        # print('current cache frequencies', v)
        # print('best not in cache', self.counts[best_not_in_cache])
        # s, _ = torch.sort(best_not_in_cache)
        # print('indices of best not in cache', s)
        
        # print(self.counts[most_common_nids])
        # print(self.counts[best_not_in_cache])

        # print('frequencies of nids being removed', self.counts[remove_nids])
        # s, _ = torch.sort(remove_nids)
        # print('indices of nids being removed:', s)

        current_worst_avg_count = torch.mean(self.counts[remove_nids]).item()
        replace_avg_count = torch.mean(self.counts[best_not_in_cache]).item()

        if current_worst_avg_count < replace_avg_count:

            self.cache_mask[remove_nids] = False
            self.cache_mapping[remove_nids] = -1

            self.cache_mapping[best_not_in_cache] = torch.arange(cache_start_idx, cache_stop_idx)
            self.reverse_mapping[cache_start_idx : cache_stop_idx] = best_not_in_cache
            self.cache_mask[best_not_in_cache]= True
            # print('Replacing', current_worst_avg_count, 'with', replace_avg_count)
        else:
            self.cache_mask[remove_nids] = True

        assert(torch.count_nonzero(self.cache_mask.int()).item() == self.cache_size)


        if self.num_updates % self.UPDATES_TO_DECREMENT == 0:
            # Divide total counts in half
            torch.div(self.counts, 2, rounding_mode='floor', out=self.counts)
            # Divide partition counts in half
            torch.div(self.hits_per_partition, 2, rounding_mode='floor', out=self.hits_per_partition)
        self.num_updates += 1
    
    def check_cache(self, request_nids) -> int:
        ''' Return how many of request_nids are in the cache (how many cache hits).
            
            Also updates the summary statistics.
        '''
        self.counts[request_nids] += 1
        in_mask = self.cache_mask[request_nids]

        cache_slots = self.cache_mapping[request_nids[in_mask]]
        assert(torch.all(cache_slots >= 0))
        self.hits_per_partition[self.partition_mapping[cache_slots]] += 1
        return torch.count_nonzero(in_mask.int())
    

class CountingCache:
    UPDATES_TO_DECREMENT = 1
    ''' LFU-like cache, can be updated with update_cache 
        Partition 0, 66.25 % neighbors in subgraph, 80.00 % seed in subgraph, 52.45 % in topk, 78.71 % in LFU, bias is 0.8                                         
        Partition 1, 59.99 % neighbors in subgraph, 80.10 % seed in subgraph, 50.14 % in topk, 73.48 % in LFU, bias is 0.8                                         
        Partition 2, 64.57 % neighbors in subgraph, 79.97 % seed in subgraph, 55.40 % in topk, 81.49 % in LFU, bias is 0.8                                         
        Partition 3, 71.17 % neighbors in subgraph, 79.99 % seed in subgraph, 51.94 % in topk, 85.09 % in LFU, bias is 0.8                                         
        Partition 4, 54.17 % neighbors in subgraph, 57.43 % seed in subgraph, 56.13 % in topk, 74.54 % in LFU, bias is 0.8                                         
    '''

    def __init__(self, init_indices, num_total_nodes, max_updates = 10000):
        # TODO figure out why changing max updates significantly affects hit rate
        self.num_total_nodes = num_total_nodes
        self.cache_size = init_indices.shape[0]
        self.cache_mask = torch.zeros(num_total_nodes, dtype=torch.bool, device=torch.device('cuda', 0))
        self.counts = torch.zeros(num_total_nodes, dtype=torch.bfloat16, device=torch.device('cuda', 0))

        self.cache_mask[init_indices] = True
        self.cache_mapping = -1 * torch.ones(num_total_nodes, device=torch.device('cuda', 0)).long()
        self.cache_mapping[init_indices] = torch.arange(self.cache_size, device=torch.device('cuda', 0))

        self.reverse_mapping = init_indices
        # self.max_updates = max_updates
        self.max_updates = self.cache_size

        self.num_updates = 0
        self.num_nodes = num_total_nodes
        self.device = torch.device('cuda', 0)
        
    def update_cache(self):
        cache_mask = self.cache_mask
        cache_mapping = self.cache_mapping
        cache_size = self.cache_size

        v, most_common_nids = torch.topk(self.counts, cache_size, sorted=True)
            
        assert(torch.all(self.counts >= 0))
        # print(most_common_nids)
        print(v[:10].min())

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
            cache_mapping[requires_update_mask] = requires_update_cache_idx

            cache_mask[:] = most_common_mask
            cache_mapping[~most_common_mask] = -1

            [torch.cuda.synchronize(i) for i in range(torch.cuda.device_count())]
            
        # torch.div(self.counts, 2, rounding_mode='floor', out=self.counts)
        self.counts /= 2
        # self.counts *= 0
        torch.cuda.synchronize(self.device)

    # def update_cache(self):
    #     assert(torch.count_nonzero(self.cache_mask.int()).item() == self.cache_size)
    #     _, worst_cache_idxs = torch.topk(self.counts[self.cache_mask], self.max_updates, largest=False, sorted=True)
    #     _, most_common_nids = torch.topk(self.counts, self.cache_size + self.max_updates, sorted=True)

    #     remove_nids = self.reverse_mapping[worst_cache_idxs]
    #     # Make it okay to keep the best parts of the worst partition
    #     # self.cache_mask[remove_nids] = False

    #     best_not_in_cache = most_common_nids[self.cache_mask[most_common_nids] == False]
    #     best_not_in_cache = best_not_in_cache[:self.max_updates]


    #     # Need to be sorted in descending order of hit frequency
    #     _, sort_idx = torch.sort(self.counts[remove_nids], descending=True)
    #     remove_nids = remove_nids[sort_idx]

    #     # print(self.counts[best_not_in_cache])
    #     # print(self.counts[remove_nids])
        
    #     n = best_not_in_cache.shape[0]
    #     assert(best_not_in_cache.shape[0] == n)
    #     assert(remove_nids.shape[0] == n)

    #     i1 = 0
    #     i2 = 0
    #     while i1 + i2 < n:
    #         if self.counts[best_not_in_cache[i1]] > self.counts[remove_nids[i2]]:
    #             # Add to cache
    #             i1 += 1    
    #         else:
    #             # Keep in cache
    #             i2 += 1

    #     best_not_in_cache = best_not_in_cache[:i1]
    #     remove_nids = remove_nids[i2:]

    #     # print(best_not_in_cache.shape, remove_nids.shape)
    #     worst_cache_idxs = self.cache_mapping[remove_nids]

    #     # Mask off to just be the ones where the update is better
    #     # i1 = 0
    #     # i2 = 0
    #     # replace_idx = n - 1
    #     # mask = []
    #     # mapping = []
    #     # while i1 < n and i2 < n:
    #     #     if self.counts[best_not_in_cache[i1]] > self.counts[remove_nids[i2]]:
    #     #         # Add element not in cache
    #     #         mask.append(True)
    #     #         mapping.append(replace_idx)
    #     #         replace_idx -= 1
    #     #     else:
    #     #         # Keep in cache
    #     #         i2 += 1
    #     #         mask.append(False)
    #     #     i1 += 1

    #     # improvement_mask = torch.tensor(mask)
    #     # # improvement_mask = self.counts[best_not_in_cache] > self.counts[remove_nids]

    #     # best_not_in_cache = best_not_in_cache[improvement_mask]
    #     # remove_nids = remove_nids[improvement_mask]

    #     # best_not_in_cache = best_not_in_cache[torch.tensor(mapping, dtype=torch.long)]
    #     # print(best_not_in_cache.shape, remove_nids.shape)
    #     # worst_cache_idxs = self.cache_mapping[remove_nids]


    #     current_worst_avg_count = torch.mean(self.counts[remove_nids]).item()
    #     replace_avg_count = torch.mean(self.counts[best_not_in_cache]).item()

    #     if current_worst_avg_count < replace_avg_count:
    #         # print('cache size', self.cache_size)
    #         # print('true in mask', torch.count_nonzero(self.cache_mask.int()).item())
    #         self.cache_mask[remove_nids] = False
    #         self.cache_mapping[remove_nids] = -1

    #         self.cache_mapping[best_not_in_cache] = worst_cache_idxs
    #         self.reverse_mapping[worst_cache_idxs] = best_not_in_cache
    #         self.cache_mask[best_not_in_cache] = True

    #         # print('finished cache', self.counts[self.reverse_mapping[worst_cache_idxs]])
    #         # print('Replacing', current_worst_avg_count, 'with', replace_avg_count)
    #         # print('true in mask', torch.count_nonzero(self.cache_mask.int()).item())


    #     assert(torch.count_nonzero(self.cache_mask.int()).item() == self.cache_size)


    #     if self.num_updates % self.UPDATES_TO_DECREMENT == 0:
    #         # Divide total counts in half
    #         torch.div(self.counts, 2, rounding_mode='floor', out=self.counts)

    #     self.num_updates += 1
    
    def check_cache(self, request_nids) -> int:
        ''' Return how many of request_nids are in the cache (how many cache hits).
            
            Also updates the summary statistics.
        '''
        request_nids = request_nids.to(self.device)
        self.counts[request_nids] += 1
        in_mask = self.cache_mask[request_nids]

        cache_slots = self.cache_mapping[request_nids[in_mask]]
        assert(torch.all(cache_slots >= 0))
        return torch.count_nonzero(in_mask.int())

def main():
    # infer_data = InferenceDataset('reddit', 0.1, verbose=True)
    infer_data = InferenceDataset('ogbn-products', 0.1, verbose=True)
    # infer_data = InferenceDataset('ogbn-papers100M', 0.01, verbose=True)
    # infer_data = InferenceDataset('ogbn-arxiv', 0.4, verbose=True)
    g = infer_data[0]

    bias = 0.8
    # trace = infer_data.create_inference_trace(trace_len=256000, subgraph_bias=bias)
    trace = infer_data.create_inference_trace(trace_len=1_000_000, subgraph_bias=bias)

    requests_per_partition = int(math.ceil(len(trace) / infer_data._num_partitions))
    print('Requests per partition:', requests_per_partition)
    part_mapping = infer_data._orig_nid_partitions

    # Check Top 20%
    out_deg = g.out_degrees()
    _, indices = torch.topk(out_deg, int(g.num_nodes() * 0.2), sorted=True)
    in_topk = torch.zeros(g.num_nodes(), dtype=torch.bool)
    in_topk[indices] = True

    # Check LFU-ish cache 
    cache = CountingCache(indices, g.num_nodes())
    k = 8
    print('LFU update frequency:', k, 'requests')

    # g = dgl.graph(g.edges(), device='cuda')
    g.create_formats_()
    g.pin_memory_()
    requests_handled = 0
    for partition in tqdm(range(infer_data._num_partitions)):
        for _ in range(1):
            neighbor_total = 0
            neighbors_in_subgraph = 0

            new_nid_total = 0
            new_nid_in_subgraph = 0

            in_topk_count = 0

            in_lfu = 0
            BATCH_SIZE = 256
            
            for i in tqdm(range(0, requests_per_partition, BATCH_SIZE), leave=False):
                cur_index = requests_per_partition * partition + i
                if cur_index >= len(trace):
                    break

                # Perform sampling
                new_nid = trace.nids[cur_index]
                new_nid = new_nid.reshape(1)
                # adj_nids = trace.edges[cur_index]["in"]
                adj_nids = trace.edges.get_batch(cur_index, cur_index + BATCH_SIZE).in_edge_endpoints
                
                frontier = dgl.sampling.sample_neighbors(g, adj_nids.to("cuda"), -1)
                all_seeds = torch.cat((adj_nids.unique(), new_nid))

                first_mfg = dgl.to_block(frontier, all_seeds) # Need to do cat here as should have target node
                first_mfg = first_mfg.to('cpu')
                
                in_partition_mask = part_mapping[first_mfg.ndata['_ID']['_N']] == partition

                # Check if sampled neighbors are in the partition or not
                # With locality we would expect it to be the case
                neighbors_in_subgraph += torch.count_nonzero(in_partition_mask.int())
                neighbor_total += first_mfg.num_src_nodes()

                # Neighbors in Top k
                in_topk_count += torch.count_nonzero(in_topk[first_mfg.ndata['_ID']['_N']].int())

                # Neighbros in LFU-ish
                in_lfu += cache.check_cache(first_mfg.ndata['_ID']['_N'])
                if (requests_handled + 1) % k == 0:
                    cache.update_cache()
                requests_handled += 1

                # Seeds In Subgraph
                if part_mapping[trace.nids[cur_index]] == partition:
                    new_nid_in_subgraph += 1
                new_nid_total += 1

            print(f'Partition {partition}, {neighbors_in_subgraph/neighbor_total * 100:.2f} % neighbors in subgraph, {new_nid_in_subgraph / new_nid_total * 100:.2f} % seed in subgraph, {in_topk_count/neighbor_total* 100:.2f} % in topk, {in_lfu/neighbor_total*100:.2f} % in LFU, bias is {bias}')

if __name__ == '__main__':
    main()