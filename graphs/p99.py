from util import load_df_throughput_p99_latency
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')


def main(dir, paths, model_name, graph_name, batch_size, file_suffix='', policy_names=None):
    rates = [50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000]
    # rates = [i for i in range(50, 801, 50)]
    dfs = []
    # if policy_names is None:
        # policy_names = ['Baseline (no cache)', 'Static (degree)', 'Counting', 'LFU', 'Async', '1% static', '1% count', 'new sampled static']
    for rate in rates:
        for i, path in enumerate(paths):
            df = load_df_throughput_p99_latency(model_name, os.path.join(dir, f'rate_{rate}', path), num_stores=2, executors_per_store=8)

            if policy_names:
                df['policy'] = policy_names[i]
            else:
                df['policy'] = path

            df['Request rate (req/s)'] = rate
            dfs.append(df)

    df = pd.concat(dfs)
    # df['total (ms)'] = df['total'] * 1000
    print(df)

    suffix = "(uniform)"
    if 'bias' in path:
        suffix = "(bias 0.8)"


    g = sns.relplot(data=df, x='Request rate (req/s)', y='P99 Latency', hue='policy', kind='line', errorbar=('ci', 80), height=4)#, aspect=0.6)
    g.set(yscale="log")
    plt.suptitle(
    f'P99 Latency {suffix} | {graph_name} batch size: {batch_size}')
    plt.tight_layout()
    plt.ylabel('P99 Latency (s)')
    plt.savefig(f'P99_latency_{model_name}{file_suffix}_gpus_{i}.png',
                bbox_inches='tight', dpi=250)
    plt.clf()


if __name__ == '__main__':
    pinned = ['pinned/']#, '']
    cache_ratios = [0.2]#, 0.1]
    for pin in pinned:
        pin_stripped = pin.replace("/", "")
        for c in cache_ratios:
            dir = 'p99_latency'

            main(dir, [
                #  f'testing/gpu/{pin}uniform/baseline',
                f'gpu/{pin}bias_0.8/static_{c}',
                f'gpu/{pin}bias_0.8/count_{c}',
                f'gpu/{pin}bias_0.8/cpp_{c}',
                f'gpu/{pin}bias_0.8/cpp_lock_{c}',
                # f'testing/gpu/{pin}uniform/lfu_{c}',
                # f'testing/gpu/{pin}uniform/hybrid_{c}',
                ], 'GCN', 'ogbn-products', 128, f'_bias_0.8_{pin_stripped}c{c}', ['Static', 'Frequency Prefetch', 'Frequency Lock-free', 'Frequency R/W Lock'])

            main(dir, [
                #  f'testing/gpu/{pin}uniform/baseline',
                f'gpu/{pin}uniform/static_{c}',
                f'gpu/{pin}uniform/count_{c}',
                f'gpu/{pin}uniform/cpp_{c}',
                f'gpu/{pin}uniform/cpp_lock_{c}',
                # f'testing/gpu/{pin}uniform/lfu_{c}',
                # f'testing/gpu/{pin}uniform/hybrid_{c}',
                ], 'GCN', 'ogbn-products', 128, f'_uniform_{pin_stripped}c{c}', ['Static', 'Frequency Prefetch', 'Frequency Lock-free', 'Frequency R/W Lock'])

