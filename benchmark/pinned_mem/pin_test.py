from fast_inference.dataset import InferenceDataset
from fast_inference.feat_server import FeatureServer
import torch
from fast_inference.timer import enable_timers, print_timer_info, Timer, clear_timers
from torch.profiler import profile, ProfilerActivity, record_function
from timeit import timeit
import pandas as pd
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torch
plt.style.use('seaborn')


def main():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    transfer_sizes = [1,
                      2,
                      4,
                      8,
                      16,
                      32,
                      64,
                      100,
                      128,
                      150,
                      175,
                      256
                      ]

    results = {'Transfer Size (MB)': [], 'Transfer Time (ms)': [], 'Transfer Type': []}

    for size in transfer_sizes:
        b = size * 1000 * 1000
        print(f'Transfering {size} MB')
        tensor = torch.rand(b // 4, dtype=torch.float32)
        # Pageable memory
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        cuda_tensor = tensor.to('cuda')
        end.record()
        pageable_time = start.elapsed_time(end)

        tensor = tensor.pin_memory()
        start.record()
        cuda_tensor = tensor.to('cuda')
        end.record()
        pinned_time = start.elapsed_time(end)

        results['Transfer Size (MB)'].append(size)
        results['Transfer Size (MB)'].append(size)
        results['Transfer Time (ms)'].append(pageable_time)
        results['Transfer Time (ms)'].append(pinned_time)
        results['Transfer Type'].append('Pageable')
        results['Transfer Type'].append('Pinned')

    df = pd.DataFrame.from_dict(results)

    # Lineplot for latency over time
    g = sns.lineplot(data=df, x='Transfer Size (MB)', y='Transfer Time (ms)', hue='Transfer Type')
    g.set_title(f'CUDA Pageable vs Pinned CPU-GPU Transfer Time')
    plt.tight_layout()
    # g.set_ylabel('Response time (s)')
    # g.set_xlabel('Time (request ID)')
    plt.savefig(f'transfer_cpu_gpu.png', bbox_inches='tight', dpi=250)
    plt.clf()


if __name__ == '__main__':
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
        main()
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=10))
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_memory_usage", row_limit=10))
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_memory_usage", row_limit=10))
    prof.export_chrome_trace("transfer_trace.json")
