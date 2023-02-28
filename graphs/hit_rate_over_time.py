from util import load_df
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')


def main(paths, policies, model_name, graph_name, batch_size):
    dfs = []
    for i, path in enumerate(paths):
        df = load_df(model_name + "_cache_info", path, graph_name, batch_size)
        df['Cache Policy'] = policies[i]
        dfs.append(df)
    cache_df = pd.concat(dfs)

    g = sns.lineplot(data=cache_df, x=cache_df.index, y='hit_rate', hue='Cache Policy')
    g.set_title(f'Cache hit rate over 5 partitions (bias 0.8) | {model_name} {graph_name} batch size: {batch_size}')
    g.set_xlabel('Time (request ID)')
    plt.tight_layout()
    plt.savefig(f'CROT_{model_name}.png', bbox_inches='tight', dpi=250)
    plt.clf()
    
if __name__ == '__main__':
    main([
          'benchmark/data/new_cache_gpu_bias_0.8',
          'benchmark/data/new_cache_gpu_bias_0.8_count',
          'benchmark/data/new_cache_gpu_bias_0.8_LFU',
          ],
           ['Static (Degree)', 'Counting', 'LFU']
           , 'GCN', 'ogbn-products', 256)