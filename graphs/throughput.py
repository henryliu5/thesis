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

    df = df.reset_index()
    num_gpus = 2
    for i in range(1, num_gpus + 1):
        plot_df = df.loc[df['num_devices'] == i]
        sns.barplot(data=plot_df, x='executors_per_store', y='throughput (req/s)', hue='policy')
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
            main([
                #  f'testing/gpu/{pin}uniform/baseline',
                f'throughput_testing/gpu/{pin}uniform/static_{c}',
                f'throughput_testing/gpu/{pin}uniform/count_{c}',
                f'throughput_testing/gpu/{pin}uniform/cpp_{c}',
                f'throughput_testing/gpu/{pin}uniform/cpp_lock_{c}',
                # f'testing/gpu/{pin}uniform/lfu_{c}',
                # f'testing/gpu/{pin}uniform/hybrid_{c}',
                ], 'GCN', 'ogbn-products', 256, f'_uniform_{pin_stripped}c{c}', ['static', 'count', 'Lock-free', 'R/W Lock'])
