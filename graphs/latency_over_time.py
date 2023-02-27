from util import load_df
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')


def main(paths, model_name, graph_name, batch_size):
    dfs = []
    for path in paths:
        df = load_df(model_name, path, graph_name, batch_size)
        df['path'] = path
        dfs.append(df)

    df = pd.concat(dfs)
    df['total (ms)'] = df['total'] * 1000
    print(df)

    # Empirical CDF
    sns.displot(data=df, kind="ecdf", x='total (ms)', hue='path')
    plt.tight_layout()
    plt.savefig(f'CDF_{model_name}.png', bbox_inches='tight', dpi=250)
    plt.clf()

    sns.lineplot(data=df, x=df.index, y='total', hue='path')
    # plt.ylim(0, 0.1)
    plt.tight_layout()
    plt.savefig(f'LOT_{model_name}.png', bbox_inches='tight', dpi=250)


    
if __name__ == '__main__':
    main(['benchmark/data/new_baseline_gpu_bias_0.8',
          'benchmark/data/new_cache_gpu_bias_0.8',
          'benchmark/data/new_cache_gpu_bias_0.8_only_0',
          'benchmark/data/new_cache_gpu_bias_0.8_only_1',
          'benchmark/data/new_cache_gpu_bias_0.8_only_2',
          'benchmark/data/new_cache_gpu_bias_0.8_count',
        #   'benchmark/data/new_cache_gpu_bias_0.8_count_divide',
        #   'benchmark/data/new_cache_gpu', 'benchmark/data/new_baseline_gpu'
          ], 'GCN', 'ogbn-products', 256)
