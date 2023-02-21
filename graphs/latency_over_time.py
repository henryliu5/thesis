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

    sns.lineplot(data=df, x=df.index, y='total', hue='path')
    plt.savefig(f'LOT_{model_name}.png', bbox_inches='tight', dpi=250)

    # Empirical CDF
    sns.displot(data=df, kind="ecdf", x='total (ms)', hue='path')
    plt.savefig(f'CDF_{model_name}.png', bbox_inches='tight', dpi=250)


if __name__ == '__main__':
    main(['benchmark/data/new_cache_gpu',
         'benchmark/data/new_cache_slow_gpu'], 'GCN', 'cora', 256)
