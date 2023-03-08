from util import load_df
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')


def main(paths, policies, model_name, graph_name, batch_size, file_suffix = ''):
    dfs = []
    for i, path in enumerate(paths):
        df = load_df(model_name + "_cache_info", path, graph_name, batch_size)
        df['Cache Policy'] = policies[i]
        dfs.append(df)
    cache_df = pd.concat(dfs)

    suffix = "(uniform sampled)"
    if 'bias' in path:
        suffix = "(5 subgraphs sampled, bias 0.8)"

    g = sns.lineplot(data=cache_df, x=cache_df.index, y='hit_rate', hue='Cache Policy')
    g.set_title(f'Cache hit rate {suffix} | {model_name} {graph_name} batch size: {batch_size}')
    g.set_xlabel('Time (request ID)')
    plt.tight_layout()
    plt.savefig(f'CROT{file_suffix}.png', bbox_inches='tight', dpi=250)
    plt.clf()

    cache_df['transfer_size (MB)'] = (cache_df['request_size'] - cache_df['cache_hits']) * 100 * 4 / 1000 / 1000

    g = sns.lineplot(data=cache_df, x=cache_df.index, y='transfer_size (MB)', hue='Cache Policy')
    g.set_title(f'Cache hit rate {suffix} | {model_name} {graph_name} batch size: {batch_size}')
    g.set_xlabel('Time (request ID)')
    plt.tight_layout()
    plt.savefig(f'Transfer_size{file_suffix}.png', bbox_inches='tight', dpi=250)
    plt.clf()
    
if __name__ == '__main__':
    main([
        #   'benchmark/data/new_cache_gpu_bias_0.8',
        #   'benchmark/data/new_cache_gpu_bias_0.8_count',
        #   'benchmark/data/new_cache_gpu_bias_0.8_LFU',
        #   'benchmark/data/new_cache_gpu_bias_0.8_hybrid',
        #   'benchmark/data/new_cache_gpu',
        #   'benchmark/data/new_cache_gpu_count',
        #   'benchmark/data/new_cache_gpu_LFU',
        #   'benchmark/data/new_cache_gpu_hybrid',

          'benchmark/fast_data/new_cache_gpu_static',
          'benchmark/fast_data/new_cache_gpu_count',
        #   'benchmark/fast_data/new_cache_gpu_lfu',
          'benchmark/fast_data/new_cache_gpu_async',
          'fast_sampling/gpu/pinned/uniform/cpp_0.2', 
        #   'benchmark/fast_data/new_cache_gpu_bias_0.8_static',
        #   'benchmark/fast_data/new_cache_gpu_bias_0.8_count',
        #   'benchmark/fast_data/new_cache_gpu_bias_0.8_lfu',
        #   'benchmark/fast_data/new_cache_gpu_bias_0.8_async', 
          ],
           ['Static (Degree)', 'Counting', 'Async', 'CPP']
           , 'GCN', 'ogbn-products', 256)

    main([
          'benchmark/fast_data/new_cache_gpu_bias_0.8_static',
          'benchmark/fast_data/new_cache_gpu_bias_0.8_count',
          'benchmark/fast_data/new_cache_gpu_bias_0.8_lfu',
          'benchmark/fast_data/new_cache_gpu_bias_0.8_async', 
          'fast_sampling/gpu/pinned/bias_0.8/cpp_0.2', 
          ],
           ['Static (Degree)', 'Counting', 'LFU', 'Async', 'CPP']
           , 'GCN', 'ogbn-products', 256, f'_biased')