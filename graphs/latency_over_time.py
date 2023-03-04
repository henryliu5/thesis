from util import load_df
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')


def main(paths, model_name, graph_name, batch_size, file_suffix = ''):
    dfs = []
    policies = ['Baseline (no cache)', 'Static (degree)', 'Counting', 'LFU', 'Async']
    for i, path in enumerate(paths):
        df = load_df(model_name, path, graph_name, batch_size)
        df['policy'] = policies[i]
        df['path'] = path
        dfs.append(df)

    df = pd.concat(dfs)
    df['total (ms)'] = df['total'] * 1000
    print(df)


    suffix = "(uniform sampled)"
    if 'bias' in path:
        suffix = "(5 subgraphs sampled, bias 0.8)"

    # Empirical CDF
    g = sns.displot(data=df, kind="ecdf", x='total (ms)', hue='policy')
    plt.suptitle(f'Request latency CDF {suffix} | {model_name} {graph_name} batch size: {batch_size}')
    plt.xlim(0, 120)
    plt.tight_layout()
    plt.savefig(f'CDF_{model_name}{file_suffix}.png', bbox_inches='tight', dpi=250)
    plt.clf()

    # Lineplot for latency over time
    g = sns.lineplot(data=df, x=df.index, y='total', hue='policy')
    g.set_title(f'Latency over time {suffix} | {model_name} {graph_name} batch size: {batch_size}')
    plt.ylim(0, 0.1)
    plt.tight_layout()
    g.set_ylabel('Response time (s)')
    g.set_xlabel('Time (request ID)')
    plt.savefig(f'LOT_{model_name}{file_suffix}.png', bbox_inches='tight', dpi=250)
    plt.clf()

if __name__ == '__main__':
    main([
        #   'benchmark/data/new_baseline_gpu_bias_0.8',
        #   'benchmark/data/new_cache_gpu_bias_0.8',
        # #   'benchmark/data/new_cache_gpu_bias_0.8_only_0',
        # #   'benchmark/data/new_cache_gpu_bias_0.8_only_1',
        # #   'benchmark/data/new_cache_gpu_bias_0.8_only_2',
        #   'benchmark/data/new_cache_gpu_bias_0.8_count',
        #   'benchmark/data/new_cache_gpu_bias_0.8_LFU',
        #   'benchmark/data/new_cache_gpu_bias_0.8_hybrid',

          'benchmark/data/new_baseline_gpu',
          'benchmark/data2/new_cache_gpu_static',
          'benchmark/data2/new_cache_gpu_count',
          'benchmark/data2/new_cache_gpu_lfu',
          'benchmark/data2/new_cache_gpu_async',

        #   'benchmark/data/new_baseline_gpu_bias_0.8',
        #   'benchmark/data2/new_cache_gpu_bias_0.8_static',
        #   'benchmark/data2/new_cache_gpu_bias_0.8_count',
        #   'benchmark/data2/new_cache_gpu_bias_0.8_lfu',
        #   'benchmark/data2/new_cache_gpu_bias_0.8_async',
          ], 'GCN', 'ogbn-products', 256)

    main([
          'benchmark/data/new_baseline_gpu_bias_0.8',
          'benchmark/data2/new_cache_gpu_bias_0.8_static',
          'benchmark/data2/new_cache_gpu_bias_0.8_count',
          'benchmark/data2/new_cache_gpu_bias_0.8_lfu',
          'benchmark/data2/new_cache_gpu_bias_0.8_async',
          ], 'GCN', 'ogbn-products', 256, '_biased')
    
    main([
          'benchmark/data/new_baseline_gpu',
          'benchmark/fast_data/new_cache_gpu_static',
          'benchmark/fast_data/new_cache_gpu_count',
          'benchmark/fast_data/new_cache_gpu_lfu',
          'benchmark/fast_data/new_cache_gpu_async',
          ], 'GCN', 'ogbn-products', 256, '_pinned')

    main([
          'benchmark/data/new_baseline_gpu_bias_0.8',
          'benchmark/fast_data/new_cache_gpu_bias_0.8_static',
          'benchmark/fast_data/new_cache_gpu_bias_0.8_count',
          'benchmark/fast_data/new_cache_gpu_bias_0.8_lfu',
          'benchmark/fast_data/new_cache_gpu_bias_0.8_async',
          ], 'GCN', 'ogbn-products', 256, '_biased_pinned')
