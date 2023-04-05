from util import load_df_cache_info
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')


def main(paths, policies, model_name, graph_name, batch_size, file_suffix = ''):
    dfs = []
    for i, path in enumerate(paths):
        df = load_df_cache_info(model_name, path, graph_name, batch_size)
        df['Cache Policy'] = policies[i]
        df['Request ID (Time)'] = df.index
        dfs.append(df)
    cache_df = pd.concat(dfs, ignore_index=True)

    suffix = "(uniform sampled)"
    if 'bias' in path:
        suffix = "(5 subgraphs sampled, bias 0.8)"

    g = sns.lineplot(data=cache_df, x='Request ID (Time)', y='hit_rate', hue='Cache Policy')
    g.set_title(f'Cache hit rate {suffix} | {model_name} {graph_name} batch size: {batch_size}')
    g.set_ylabel('Hit Ratio')
    plt.tight_layout()
    plt.savefig(f'CROT{file_suffix}.png', bbox_inches='tight', dpi=250)
    plt.clf()

    cache_df['transfer_size (MB)'] = (cache_df['request_size'] - cache_df['cache_hits']) * 100 * 4 / 1000 / 1000

    g = sns.lineplot(data=cache_df, x='Request ID (Time)', y='transfer_size (MB)', hue='Cache Policy')
    g.set_title(f'Cache hit rate {suffix} | {model_name} {graph_name} batch size: {batch_size}')
    plt.tight_layout()
    plt.savefig(f'Transfer_size{file_suffix}.png', bbox_inches='tight', dpi=250)
    plt.clf()
    
if __name__ == '__main__':
    cache_ratios = [0.2]#, 0.1]
    # cache_ratios = [0.05, 0.025]
    for c in cache_ratios:
        main([
            f'testing/gpu/pinned/uniform/static_{c}',
            f'testing/gpu/pinned/uniform/count_{c}',
            f'testing/gpu/pinned/uniform/cpp_{c}', 
            # f'testing/gpu/pinned/uniform/cpp_lock_{c}', 
            # f'testing/gpu/pinned/uniform/cpp_{c}', 
            # f'testing/gpu/pinned/uniform/lfu_{c}',
            ],
            [f'Static {c*100}%', f'Full Update (Frequency) {c*100}%', f'Incremental Update (Frequency) {c*100}%', 'LFU'],
            'GCN', 'ogbn-products', 256, f'c{c}')
        main([
            f'throughput_testing/gpu/pinned/bias_0.8/static_{c}',
            f'throughput_testing/gpu/pinned/bias_0.8/count_{c}',
            f'throughput_testing/gpu/pinned/bias_0.8/cpp_{c}', 
            # f'testing/gpu/pinned/bias_0.8/cpp_lock_{c}', 
            ],
            [f'Static {c*100}%', f'Full Update (Frequency) {c*100}%', f'Incremental Update (Frequency) {c*100}%', 'LFU'],
            'GCN', 'ogbn-products', 256, f'_biased_c{c}')