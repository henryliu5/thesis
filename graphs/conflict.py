import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import glob
plt.style.use('seaborn')


def main(paths, policies, model_name, graph_name, batch_size, file_suffix = ''):
    dfs = []
    for i, path in enumerate(paths):
        df = load_df_cache_info(model_name, path, graph_name, batch_size, trials=10)
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
    cache_type = 'cpp_lock'

    files = glob.glob(f'pipeline_conflicts/{cache_type}-*-*-*-*.csv')
    dfs = []    
    for file in files:
        dfs.append(pd.read_csv(file))

    
    df = pd.concat(dfs, ignore_index=True)
    df = df[df.executors_per_store != 16]

    g = sns.displot(data=df, x='wait_time', kind="ecdf", col='num_stores', hue='executors_per_store')
    plt.tight_layout()
    plt.savefig(f'Lock_Conflicts.png', bbox_inches='tight', dpi=250)
    plt.clf()
    print(df)