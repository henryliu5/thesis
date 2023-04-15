from util import load_df_throughput
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')


def main(paths, model_name, graph_name, batch_size, file_suffix='', policy_names=None):
    dfs = []
    # if policy_names is None:
        # policy_names = ['Baseline (no cache)', 'Static (degree)', 'Counting', 'LFU', 'Async', '1% static', '1% count', 'new sampled static']
    for i, path in enumerate(paths):
        df = load_df_throughput(model_name, path, graph_name, batch_size)

        if policy_names:
            df['policy'] = policy_names[i]
        else:
            df['policy'] = path

    #     df['path'] = path
        dfs.append(df)
    df = pd.concat(dfs)
    # df['total (ms)'] = df['total'] * 1000
    print(df)

    suffix = "(uniform sampled)"
    if 'bias' in path:
        suffix = "(5 subgraphs sampled, bias 0.8)"

    df = df.reset_index()
    num_gpus = 2
    for i in range(2, 3):
        plot_df = df.loc[df['num_devices'] == i]
        g = sns.barplot(data=plot_df, x='executors_per_store', y='throughput (req/s)', hue='policy', errorbar='pi')
        g.set_title(
        f'Throughput {suffix} | {model_name} {graph_name} batch size: {batch_size}')
        plt.tight_layout()
        plt.savefig(f'throughput_{model_name}{file_suffix}_gpus_{i}.png',
                    bbox_inches='tight', dpi=250)
        plt.clf()


if __name__ == '__main__':
    # main([
    #      'fast_sampling/gpu/bias_0.8/baseline',
    #      'fast_sampling/gpu/bias_0.8/count_0.1',
    #      'fast_sampling/gpu/bias_0.8/static_0.1'], 'GCN', 'ogbn-products', 256, '_fast_sampling')#, ['count', 'static'])
    pinned = ['pinned/']#, '']
    cache_ratios = [0.2]#, 0.1]
    # cache_ratios = [0.05, 0.025]
    for pin in pinned:
        pin_stripped = pin.replace("/", "")
        for c in cache_ratios:
            # dir = 'throughput_testing'
            # dir = 'only_pin_no_thread_reduce'
            # dir = 'throughput_direct'
            # dir = 'throughput_pin_numa_cpu_0_reduce_per_executor'
            # dir = 'throughput_pin_numa_cpu_0_reduce_per_executor_with_min'
            # dir = 'throughput_pin_numa_cpu_0_reduce_by_engines'
            dir = 'multiple_throughput'
            main([
                #  f'testing/gpu/{pin}uniform/baseline',
                f'{dir}/gpu/{pin}bias_0.8/static_{c}',
                f'{dir}/gpu/{pin}bias_0.8/count_{c}',
                f'{dir}/gpu/{pin}bias_0.8/cpp_{c}',
                f'{dir}/gpu/{pin}bias_0.8/cpp_lock_{c}',
                # f'testing/gpu/{pin}bias_0.8/lfu_{c}',
                # f'testing/gpu/{pin}bias_0.8/hybrid_{c}',
                ], 'GCN', 'ogbn-products', 128, f'_bias_0.8_{pin_stripped}c{c}', ['Static', 'Frequency Prefetch', 'Frequency Lock-free', 'Frequency R/W Lock'])

            main([
                #  f'testing/gpu/{pin}uniform/baseline',
                f'{dir}/gpu/{pin}uniform/static_{c}',
                f'{dir}/gpu/{pin}uniform/count_{c}',
                f'{dir}/gpu/{pin}uniform/cpp_{c}',
                f'{dir}/gpu/{pin}uniform/cpp_lock_{c}',
                # f'testing/gpu/{pin}uniform/lfu_{c}',
                # f'testing/gpu/{pin}uniform/hybrid_{c}',
                ], 'GCN', 'ogbn-products', 128, f'_uniform_{pin_stripped}c{c}', ['Static', 'Frequency Prefetch', 'Frequency Lock-free', 'Frequency R/W Lock'])
