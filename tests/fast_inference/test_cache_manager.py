# import torch
# import fast_inference_cpp
# import time

# def test_cache_thread_counting():
#     num_total_nodes = 100
#     cache_manager = fast_inference_cpp.CacheManager(num_total_nodes, 10, 10, 10)

#     # TODO support non-unique neighborhoods if necessary
#     neighborhood = 99
#     rand_nids = torch.randint(0, num_total_nodes, (neighborhood,))
#     print('sending nids')
    
#     start_time = time.perf_counter()
#     cache_manager.incr_counts(rand_nids)
#     end_time = time.perf_counter()
#     print('send elapsed', end_time - start_time)

#     start_time = time.perf_counter()
#     cache_manager.wait_for_queue()
#     end_time = time.perf_counter()
#     print('wait for receive elapsed', end_time - start_time)

#     counts = cache_manager.get_counts()
#     print('received counts', counts)
    
#     expected_counts = torch.zeros(num_total_nodes)
#     expected_counts[rand_nids] += 1

#     assert (torch.equal(counts, expected_counts))

# def test_cache_stats():
#     num_total_nodes = 100
#     feature_dim = 10
#     cache_size = 5
#     update_frequency = 10
#     staging_area_size = 2
#     cache_manager = fast_inference_cpp.CacheManager(num_total_nodes, cache_size, update_frequency, staging_area_size)

#     graph_features = torch.randint(0, 1000, (num_total_nodes, feature_dim))
#     cache_mask = torch.zeros(num_total_nodes, dtype=torch.bool)
#     cache_mask[0:5] = True

#     cache_mapping = torch.zeros(num_total_nodes, dtype=torch.long)
#     cache_mapping[0:5] = torch.arange(5)

#     reverse_mapping = torch.arange(5)

#     cache = graph_features[0:5].to('cuda')

#     cache_manager.set_cache(graph_features, cache_mask, cache_mapping, reverse_mapping, cache)

#     for i in range(num_total_nodes):
#         x = torch.arange(i)
#         cache_manager.incr_counts(x)
        

#     start_time = time.perf_counter()
#     cache_manager.wait_for_queue()
#     end_time = time.perf_counter()
#     print('wait for receive elapsed', end_time - start_time)

#     least_used_idx = cache_manager.get_least_used_cache_indices(3)
#     print('least_used_idx', least_used_idx)

#     most_used_nids = cache_manager.get_most_common_nodes_not_in_cache(3)
#     print('most_used_nids', most_used_nids)

#     assert (torch.equal(least_used_idx.sort()[0], torch.tensor([2, 3, 4])))
#     assert (torch.equal(most_used_nids.sort()[0], torch.arange(5, 8)))

# if __name__ == '__main__':
#     # test_cache_thread_counting()
#     test_cache_stats()