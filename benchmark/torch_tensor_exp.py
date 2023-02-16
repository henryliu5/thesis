from fast_inference.dataset import InferenceDataset
from fast_inference.feat_server import FeatureServer
import torch
from fast_inference.timer import enable_timers, print_timer_info, Timer, clear_timers
from torch.profiler import profile, ProfilerActivity, record_function
from timeit import timeit
import time
import gc
# def timeit_test():
#     setup = '''
# import torch
# torch.manual_seed(0)
# torch.use_deterministic_algorithms(True)
# n = 111059956
# feats = torch.ones((n, 128))
# zero_cache_request = torch.randint(0, n, (124364,))
#     '''
#     print(timeit('feats[zero_cache_request]', setup, number=1000) / 1000)

@torch.inference_mode()
def main():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    n = 111059956
    
    infer_data = InferenceDataset('ogbn-papers100M', 0.1, force_reload=False, verbose=True)
    g = infer_data[0]

    # feats = torch.ones((n, 128))
    feats = g.ndata['feat']
    # feat_server = FeatureServer(g, 'cuda')
    # print('feats shape', feats.shape)
    # Cache 0% 70k-120k
    # Cache 20% 20k~
    # nid_is_on_gpu = torch.zeros(g.num_nodes()).bool()
    
    # # !! important line?
    with record_function('calc degrees'):
        in_deg = g.in_degrees()
        print(in_deg.shape)
        out_deg = g.out_degrees()
        print(out_deg.shape)
    
    # with record_function('del degrees'):
    #     del in_deg
    #     del out_deg
    #     print('deleting')
    
    # with record_function('gc'):
    #     gc.collect()
    #     print('collecting gc')

    # in_deg = torch.ones((n,))
    # out_deg = torch.ones((n,))
    for i in range(1000):
        zero_cache_request = torch.randint(0, n, (124364,))
        # d = feat_server.get_features(zero_cache_request, feats=['feat'])
        with Timer('allocation'):
            res = torch.empty((124364, 128), dtype=torch.float)
        with Timer('index select'):
            torch.index_select(feats, 0, zero_cache_request, out=res)
        with record_function('rand op'):
            res *= 1000
        # with Timer('MY INDEX:'):
            # x = feats[required_feats]
    # gpu_mask = nid_is_on_gpu[zero_cache_request]
    # cpu_mask = ~gpu_mask

    # with Timer('compute cpu mask'):
    #     m = zero_cache_request[cpu_mask]

    # with Timer('zero cache index feats'):
    #     x = feats[m]

    # twenty_cache_mask = torch.randint(0, n, (20_000,))
    # with Timer('twenty cache index feats', track_cuda=True):
    #     y = feats[twenty_cache_mask]
    # print(x.shape, x.device)#, y.shape, x.device)

if __name__ == '__main__':
    # timeit_test()
    enable_timers()
    clear_timers()
    # with profile() as prof:
        # main()
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
    #     main()
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=10))
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_memory_usage", row_limit=10))
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_memory_usage", row_limit=10))
    # prof.export_chrome_trace("trace.json")
    main()
    print_timer_info(2)
