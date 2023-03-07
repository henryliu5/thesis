import torch
import fast_inference_cpp
import time


def test_start_thread():
    num_total_nodes = 100
    cache_manager = fast_inference_cpp.CacheManager(num_total_nodes, 10)

    rand_nids = torch.randint(0, 1000, (num_total_nodes,))
    print('sending nids')
    
    start_time = time.perf_counter()
    cache_manager.receive_counts(rand_nids)
    end_time = time.perf_counter()
    print('send elapsed', end_time - start_time)
    print('received nids')

if __name__ == '__main__':
    test_start_thread()