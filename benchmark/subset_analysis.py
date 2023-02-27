''' Utiliy to check the hit rate of static cache'''
from fast_inference.dataset import InferenceDataset
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import math
from tqdm import tqdm
import dgl
import torch


class CountingCache:
    ''' LFU-like cache, can be updated with update_cache '''
    def __init__(self, init_indices, num_total_nodes):
        self.num_total_nodes = num_total_nodes
        self.cache_size = init_indices.shape[0]
        self.cache_mask = torch.zeros(num_total_nodes, dtype=torch.bool)
        self.counts = torch.zeros(num_total_nodes)

        self.cache_mask[init_indices] = True

    def update_cache(self):
        # Resets cache mask (nothing stored anymore)
        self.cache_mask = torch.zeros(self.num_total_nodes, dtype=torch.bool)
        _, most_common_nids = torch.topk(self.counts, self.cache_size, sorted=False)
        # Updates to most common in based on self.counts
        self.cache_mask[most_common_nids] = True
        self.counts *= 0
    
    def check_cache(self, request_nids) -> int:
        ''' Return how many of request_nids are in the cache (how many cache hits).
            
            Also updates the summary statistics.
        '''
        self.counts[request_nids] += 1
        in_mask = self.cache_mask[request_nids]
        return torch.count_nonzero(in_mask.int())
    

def main():
    # infer_data = InferenceDataset('reddit', 0.1, verbose=True)
    infer_data = InferenceDataset('ogbn-products', 0.1, verbose=True)
    g = infer_data[0]

    bias = 0.8
    trace = infer_data.create_inference_trace(trace_len=256000, subgraph_bias=bias)

    requests_per_partition = int(math.ceil(len(trace) / infer_data._num_partitions))
    print('Requests per partition:', requests_per_partition)
    part_mapping = infer_data._orig_nid_partitions

    # Check Top 20%
    out_deg = g.out_degrees()
    _, indices = torch.topk(out_deg, int(g.num_nodes() * 0.2), sorted=False)
    in_topk = torch.zeros(g.num_nodes(), dtype=torch.bool)
    in_topk[indices] = True

    # Check LFU-ish cache 
    cache = CountingCache(indices, g.num_nodes())
    k = 500
    print('LFU update frequency:', k, 'requests')

    g = dgl.graph(g.edges(), device='cuda')

    for partition in range(infer_data._num_partitions):
        neighbor_total = 0
        neighbors_in_subgraph = 0

        new_nid_total = 0
        new_nid_in_subgraph = 0

        in_topk_count = 0

        in_lfu = 0

        for i in range(requests_per_partition):
            cur_index = requests_per_partition * partition + i
            if cur_index >= len(trace):
                break

            # Perform sampling
            new_nid = trace.nids[cur_index]
            new_nid = new_nid.reshape(1)
            adj_nids = trace.edges[cur_index]["in"]
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
            if (cur_index + 1) % k == 0:
                cache.update_cache()

            # Seeds In Subgraph
            if part_mapping[trace.nids[cur_index]] == partition:
                new_nid_in_subgraph += 1
            new_nid_total += 1

        print(f'Partition {partition}, {neighbors_in_subgraph/neighbor_total * 100:.2f} % neighbors in subgraph, {new_nid_in_subgraph / new_nid_total * 100:.2f} % seed in subgraph, {in_topk_count/neighbor_total* 100:.2f} % in topk, {in_lfu/neighbor_total*100:.2f} % in LFU, bias is {bias}')

if __name__ == '__main__':
    main()