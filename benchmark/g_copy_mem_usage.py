from fast_inference.dataset import InferenceDataset
from fast_inference.models.gcn import GCN
from fast_inference.timer import enable_timers, Timer, print_timer_info
import dgl
import torch
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
from dgl.dataloading import MultiLayerFullNeighborSampler

device = 'cuda'

if __name__ == '__main__':
    infer_data = InferenceDataset('ogbn-papers100M', 0.1, force_reload=False, verbose=True)
    cpu_g = infer_data[0]
    # Start torch profiling after loading dataset
    with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
        with record_function("graph.to"):
            # This causes a GPU memory allocation and transfer of metadata - but not features
            g = cpu_g.to('cuda')

        with record_function("load 100"):
            # The first call to g.ndata copies ALL features to the GPU
            print(g.ndata['feat'][:100])
        # Subsequent calls will be much faster and not require any transfer
        with record_function("load 1"):
            print(g.ndata['feat'][300])
        # ... even a really large access, since all features are already on GPU
        with record_function("load all"):
            print(g.ndata['feat'])

        # with record_function("load_mfg"):
        #     batch_size = 128
        #     s = MultiLayerFullNeighborSampler(2)
        #     seed_nodes, output_nodes, blocks = s.sample_blocks(cpu_g, torch.arange(batch_size))
        #     blocks = [b.to(device) for b in blocks]
        #     print([b.ndata['feat'] for b in blocks])


    print(prof.key_averages(group_by_stack_n=5).table(sort_by="cuda_time_total", row_limit=10))
    print(prof.key_averages(group_by_stack_n=5).table(sort_by="cuda_memory_usage", row_limit=10))
    print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_memory_usage", row_limit=10))
    prof.export_chrome_trace("trace.json")

# Profiling output on mew0 2/8/22 NVIDIA A100 80GB:

# STAGE:2023-02-09 01:36:29 1710824:1710824 ActivityProfilerController.cpp:300] Completed Stage: Collection
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                        Memcpy HtoD (Pageable -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us       10.675s       100.00%       10.675s        3.558s           0 b           0 b           0 b           0 b             3  
#                                                load 100         0.01%       1.845ms        49.78%        6.400s        6.400s       0.000us         0.00%        6.340s        6.340s          -4 b        -436 b      52.96 Gb     -11.00 Kb             1  
#                                          aten::_to_copy         0.00%     172.000us        49.71%        6.392s     220.413ms       0.000us         0.00%        6.340s     218.626ms         800 b          84 b      52.96 Gb           0 b            29  
#                                             aten::copy_         0.00%     236.000us        49.32%        6.341s     218.649ms        6.340s        59.39%        6.340s     218.626ms          48 b          48 b           0 b           0 b            29  
#                                                aten::to         0.00%     166.000us        49.71%        6.392s     182.630ms       0.000us         0.00%        6.340s     181.147ms         788 b          48 b      52.96 Gb           0 b            35  
#                                                graph.to         0.19%      24.729ms        50.17%        6.451s        6.451s       0.000us         0.00%        6.340s        6.340s          -4 b        -268 b      19.50 Gb      19.50 Gb             1  
#                                 cudaGetDeviceProperties         0.02%       2.900ms         0.02%       2.900ms       1.450ms        6.340s        59.39%        6.340s        3.170s           0 b           0 b           0 b           0 b             2  
#                               aten::_local_scalar_dense         0.01%     656.000us         0.03%       3.244ms      15.448us     210.000us         0.00%     254.000us       1.210us           0 b           0 b           0 b           0 b           210  
#                        Memcpy DtoH (Device -> Pageable)         0.00%       0.000us         0.00%       0.000us       0.000us     239.000us         0.00%     239.000us       1.004us           0 b           0 b           0 b           0 b           238  
#                                              aten::item         0.00%     470.000us         0.03%       3.381ms      16.100us       0.000us         0.00%     231.000us       1.100us           0 b           0 b           0 b           0 b           210  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
# Self CPU time total: 12.857s
# Self CUDA time total: 10.676s

# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                                aten::to         0.00%     166.000us        49.71%        6.392s     182.630ms       0.000us         0.00%        6.340s     181.147ms         788 b          48 b      52.96 Gb           0 b            35  
#                                          aten::_to_copy         0.00%     172.000us        49.71%        6.392s     220.413ms       0.000us         0.00%        6.340s     218.626ms         800 b          84 b      52.96 Gb           0 b            29  
#                                     aten::empty_strided         0.00%      91.000us         0.40%      50.979ms       1.758ms       0.000us         0.00%       0.000us       0.000us         668 b         668 b      52.96 Gb      52.96 Gb            29  
#                                                load 100         0.01%       1.845ms        49.78%        6.400s        6.400s       0.000us         0.00%        6.340s        6.340s          -4 b        -436 b      52.96 Gb     -11.00 Kb             1  
#                                                graph.to         0.19%      24.729ms        50.17%        6.451s        6.451s       0.000us         0.00%        6.340s        6.340s          -4 b        -268 b      19.50 Gb      19.50 Gb             1  
#                                               aten::cat         0.00%     250.000us         0.01%       1.359ms      97.071us      48.000us         0.00%      48.000us       3.429us           0 b           0 b       7.00 Kb       7.00 Kb            14  
#                                           aten::resize_         0.00%      46.000us         0.00%      46.000us       3.833us       0.000us         0.00%       0.000us       0.000us           0 b           0 b       6.50 Kb       6.50 Kb            12  
#                                               aten::abs         0.00%     152.000us         0.00%     463.000us      38.583us      14.000us         0.00%      28.000us       2.333us           0 b           0 b       6.00 Kb           0 b            12  
#                                                aten::ne         0.00%     143.000us         0.00%     197.000us      21.889us      19.000us         0.00%      22.000us       2.444us           0 b           0 b       4.50 Kb       4.50 Kb             9  
#                                             aten::empty         0.00%     111.000us         0.00%     111.000us       4.826us       0.000us         0.00%       0.000us       0.000us         808 b         808 b       3.00 Kb       3.00 Kb            23  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
# Self CPU time total: 12.857s
# Self CUDA time total: 10.676s

# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                     aten::empty_strided         0.00%      91.000us         0.40%      50.979ms       1.758ms       0.000us         0.00%       0.000us       0.000us         668 b         668 b      52.96 Gb      52.96 Gb            29  
#                                                graph.to         0.19%      24.729ms        50.17%        6.451s        6.451s       0.000us         0.00%        6.340s        6.340s          -4 b        -268 b      19.50 Gb      19.50 Gb             1  
#                                               aten::cat         0.00%     250.000us         0.01%       1.359ms      97.071us      48.000us         0.00%      48.000us       3.429us           0 b           0 b       7.00 Kb       7.00 Kb            14  
#                                           aten::resize_         0.00%      46.000us         0.00%      46.000us       3.833us       0.000us         0.00%       0.000us       0.000us           0 b           0 b       6.50 Kb       6.50 Kb            12  
#                                                aten::ne         0.00%     143.000us         0.00%     197.000us      21.889us      19.000us         0.00%      22.000us       2.444us           0 b           0 b       4.50 Kb       4.50 Kb             9  
#                                             aten::empty         0.00%     111.000us         0.00%     111.000us       4.826us       0.000us         0.00%       0.000us       0.000us         808 b         808 b       3.00 Kb       3.00 Kb            23  
#                                                aten::gt         0.00%      80.000us         0.00%     113.000us      22.600us      11.000us         0.00%      12.000us       2.400us           0 b           0 b       2.50 Kb       2.50 Kb             5  
#                                                aten::eq         0.00%      44.000us         0.00%      62.000us      20.667us       6.000us         0.00%       6.000us       2.000us           0 b           0 b       1.50 Kb       1.50 Kb             3  
#                                               aten::mul         0.00%      57.000us         0.00%      78.000us      26.000us       6.000us         0.00%       6.000us       2.000us           0 b           0 b       1.50 Kb       1.50 Kb             3  
#                                       aten::bitwise_and         0.00%      59.000us         0.00%      81.000us      27.000us       7.000us         0.00%       7.000us       2.333us           0 b           0 b       1.50 Kb       1.50 Kb             3  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
# Self CPU time total: 12.857s
# Self CUDA time total: 10.676s

# STAGE:2023-02-09 01:36:29 1710824:1710824 output_json.cpp:417] Completed Stage: Post Processing