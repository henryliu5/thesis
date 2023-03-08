from util import load_df
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
        df = load_df(model_name, path, graph_name, batch_size)

        if policy_names:
            df['policy'] = policy_names[i]
        else:
            df['policy'] = path

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
    plt.suptitle(
        f'Request latency CDF {suffix} | {model_name} {graph_name} batch size: {batch_size}')
    plt.xlim(0, 120)
    plt.tight_layout()
    plt.savefig(f'CDF_{model_name}{file_suffix}.png',
                bbox_inches='tight', dpi=250)
    plt.clf()

    # Lineplot for latency over time
    g = sns.lineplot(data=df, x=df.index, y='total', hue='policy')
    g.set_title(
        f'Latency over time {suffix} | {model_name} {graph_name} batch size: {batch_size}')
    plt.ylim(0, 0.1)
    plt.tight_layout()
    g.set_ylabel('Response time (s)')
    g.set_xlabel('Time (request ID)')
    plt.savefig(f'LOT_{model_name}{file_suffix}.png',
                bbox_inches='tight', dpi=250)
    plt.clf()


if __name__ == '__main__':
    # main([
    #     #   'benchmark/data/new_baseline_gpu_bias_0.8',
    #     #   'benchmark/data/new_cache_gpu_bias_0.8',
    #     # #   'benchmark/data/new_cache_gpu_bias_0.8_only_0',
    #     # #   'benchmark/data/new_cache_gpu_bias_0.8_only_1',
    #     # #   'benchmark/data/new_cache_gpu_bias_0.8_only_2',
    #     #   'benchmark/data/new_cache_gpu_bias_0.8_count',
    #     #   'benchmark/data/new_cache_gpu_bias_0.8_LFU',
    #     #   'benchmark/data/new_cache_gpu_bias_0.8_hybrid',

    #       'benchmark/data/new_baseline_gpu',
    #       'benchmark/data3/new_cache_gpu_static',
    #       'benchmark/data3/new_cache_gpu_count',
    #       'benchmark/data3/new_cache_gpu_lfu',
    #       'benchmark/data3/new_cache_gpu_async',
    #       'benchmark/data_cache_1/new_cache_gpu_static',
    #       'benchmark/data_cache_1/new_cache_gpu_count',
    #       'new_sampling_gpu_static'
    #       ], 'GCN', 'ogbn-products', 256)

    # main([
    #       'benchmark/data/new_baseline_gpu_bias_0.8',
    #       'benchmark/data3/new_cache_gpu_bias_0.8_static',
    #       'benchmark/data3/new_cache_gpu_bias_0.8_count',
    #       'benchmark/data3/new_cache_gpu_bias_0.8_lfu',
    #       'benchmark/data3/new_cache_gpu_bias_0.8_async',
    #       'benchmark/data_cache_1/new_cache_gpu_bias_0.8_static',
    #       'benchmark/data_cache_1/new_cache_gpu_bias_0.8_count',
    #       ], 'GCN', 'ogbn-products', 256, '_biased')

    # main([
    #       'benchmark/data/new_baseline_gpu',
    #       'benchmark/data3_pinned/new_cache_gpu_static',
    #       'benchmark/data3_pinned/new_cache_gpu_count',
    #       'benchmark/data3_pinned/new_cache_gpu_lfu',
    #       'benchmark/data3_pinned/new_cache_gpu_async',
    #       'benchmark/data_cache_1_pinned/new_cache_gpu_static',
    #       'benchmark/data_cache_1_pinned/new_cache_gpu_count',
    #       ], 'GCN', 'ogbn-products', 256, '_pinned')

    # main([
    #       'benchmark/data/new_baseline_gpu_bias_0.8',
    #       'benchmark/data3_pinned/new_cache_gpu_bias_0.8_static',
    #       'benchmark/data3_pinned/new_cache_gpu_bias_0.8_count',
    #       'benchmark/data3_pinned/new_cache_gpu_bias_0.8_lfu',
    #       'benchmark/data3_pinned/new_cache_gpu_bias_0.8_async',
    #       'benchmark/data_cache_1_pinned/new_cache_gpu_bias_0.8_static',
    #       'benchmark/data_cache_1_pinned/new_cache_gpu_bias_0.8_count',
    #       ], 'GCN', 'ogbn-products', 256, '_biased_pinned')

    main([
         'fast_sampling/gpu/bias_0.8/baseline',
         'fast_sampling/gpu/bias_0.8/count_0.1',
         'fast_sampling/gpu/bias_0.8/static_0.1'], 'GCN', 'ogbn-products', 256, '_fast_sampling')#, ['count', 'static'])
    
    main([
         'fast_sampling/gpu/pinned/bias_0.8/baseline',
        #  'fast_sampling/gpu/pinned/bias_0.8/count_0.1',
        #  'fast_sampling/gpu/pinned/bias_0.8/static_0.1',
         'fast_sampling/gpu/pinned/bias_0.8/count_0.2',
         'fast_sampling/gpu/pinned/bias_0.8/static_0.2',
         'fast_sampling/gpu/pinned/bias_0.8/async_0.2',
         'fast_sampling/gpu/pinned/bias_0.8/cpp_0.2',
        #  'fast_sampling/gpu/pinned/bias_0.8/count_0.3',
        #  'fast_sampling/gpu/pinned/bias_0.8/static_0.3',
        #  'fast_sampling/gpu/pinned/bias_0.8/async_0.3'
         ], 'GCN', 'ogbn-products', 256, '_fast_sampling_pinned')#, ['count', 'static'])

    main([
         'fast_sampling/gpu/pinned/uniform/baseline',
        #  'fast_sampling/gpu/pinned/uniform/count_0.1',
        #  'fast_sampling/gpu/pinned/uniform/static_0.1',
        #  'fast_sampling/gpu/pinned/uniform/count_0.2',
         'fast_sampling/gpu/pinned/uniform/static_0.2',
        #  'fast_sampling/gpu/pinned/uniform/async_0.2',
         'fast_sampling/gpu/pinned/uniform/cpp_0.2',
        #  'fast_sampling/gpu/pinned/uniform/count_0.3',
        #  'fast_sampling/gpu/pinned/uniform/static_0.3',
        #  'fast_sampling/gpu/pinned/uniform/async_0.3'
         ], 'GCN', 'ogbn-products', 256, '_uniform_pinned')#, ['count', 'static'])
