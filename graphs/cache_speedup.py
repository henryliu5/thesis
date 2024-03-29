import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from util import load_dfs

def plot_speedup(model_name, cache_path, baseline_path, title = ""):
    graph_names = ['reddit', 'yelp', 'ogbn-products', 'ogbn-papers100M']
    batch_sizes = [32, 64, 128, 256, 512]

    baseline_df = load_dfs(model_name, baseline_path, graph_names, batch_sizes).groupby(['name', 'batch_size']).median()
    cache_df = load_dfs(model_name, cache_path, graph_names, batch_sizes).groupby(['name', 'batch_size']).median()
    # cache_df = cache_df.drop(columns=['weird index', 'compute gpu/cpu mask', 'get_features()', 'mask cpu feats', 'allocate res tensor'])
    print(baseline_df)
    print(cache_df)
    # Compute speedup
    df = baseline_df / cache_df
    df = df.dropna(axis=1)
    # Subtract 1 since the graph will place bars at 1
    print('------------------')
    print(df)
    df -= 1
    df = df.reset_index()
    df['name'] = pd.Categorical(df['name'], graph_names)
    df['dataset'] = df['name']
    df['Request Batch Size'] = df['batch_size']
    df['Data Loading'] = df['dataloading']
    df.sort_values('name')
    print(df)

    runtime_parts = ['sampling', 'feature gather', 'CPU-GPU copy', 'model', 'total', 'Data Loading']
    for p in runtime_parts:
        g = sns.catplot(df, x='Request Batch Size', y=p,
                        col='dataset', kind="bar", col_wrap=2, bottom=1, height=3)
        
        g.fig.subplots_adjust(top=.9)
        g.fig.suptitle(f'{model_name} {p} speedup {title}', fontsize=12)
        # iterate through axes
        for ax in g.axes.ravel():
            # add annotations
            for c in ax.containers:
                labels = [f'{(1 + v.get_height()):.3f}' for v in c]
                ax.bar_label(c, labels=labels, padding=8)
            ax.margins(y=0.2)

        plt.savefig(f'speedup_{model_name}_{p}.png', dpi=250)
        plt.clf()

if __name__ == '__main__':
    cache_path = 'testing/gpu/pinned/uniform/cpp_0.2'
    baseline_path = 'testing/gpu/pinned/uniform/static_0.2'
    plot_speedup('GCN', cache_path, baseline_path, title="| Uniform Requests, Frequency Lock-Free vs. Static Cache")

    # cache_path = 'testing/gpu/pinned/bias_0.8/cpp_0.2'
    # baseline_path = 'testing/gpu/pinned/bias_0.8/static_0.2'
    # plot_speedup('GCN', cache_path, baseline_path, title="| Biased Requests, Frequency Lock-Free vs. Static Cache")

