import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pickle

def simple_stats(graph_name, batch_size):
    infile = open(f'{graph_name}_{batch_size}', 'rb')
    nodes = pickle.load(infile)
    infile.close()
    sizes = [len(x) for x in nodes]

    # sizes = np.load(f'{graph_name}_{batch_size}.npz')['arr_0']
    var = np.var(sizes)
    # print(len(sizes), 'variance: ', var)

    df = pd.DataFrame(sizes, columns=['mfg size'])
    sns.displot(df, kde=True)
    plt.show()
    plt.savefig(f'{graph_name}_{batch_size}_stats.png')

def cache_hit_plots():
    cache_sizes = {'ogbn-arxiv': [0.0005, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02],
                   'ogbn-products': [0.01, 0.03, 0.05, 0.08, 0.11, 0.15, 0.2],
                   'ogbn-products-METIS': [0.005, 0.01, 0.02, 0.04, 0.08, 0.10],
                   'reddit': [0.0025, 0.005, 0.01, 0.02, 0.04, 0.06]}
    cache_policies = ['lru', 'mine', 'fifo', 'static']
    datasets = ['ogbn-arxiv', 'ogbn-products', 'ogbn-products-METIS', 'reddit']
    graph_info = {'ogbn-arxiv': '169,343 Nodes, 1,166,243 Edges',
                   'ogbn-products': '2,449,029 Nodes, 61,859,140 Edges',
                   'ogbn-products-METIS': '1/5 of ogbn-products',
                   'reddit': '232,965 Nodes, 114,615,892 Edges'}

    for graph_name in datasets:
        df = pd.DataFrame()
        policy_type = []
        hit_ratios = []
        pd_cache_sizes = []
        for policy in cache_policies:
            for cache_size in cache_sizes[graph_name]:
                infile = open(f'results/{policy}_hit_ratio_cache_{cache_size}_{graph_name}.txt', 'r')
                hit_ratio = float(infile.read())
                infile.close()
                hit_ratios.append(hit_ratio)
                policy_type.append(policy)
                pd_cache_sizes.append(cache_size)

        df['hit_rate'] = hit_ratios
        df['policy'] = policy_type
        df['cache_size'] = pd_cache_sizes
        print(df)
        sns.lineplot(df, x='cache_size', y='hit_rate', hue='policy')
        plt.title(f'Cache hit rate - {graph_name} - {graph_info[graph_name]}')
        plt.savefig(f'cache_hit_{graph_name}.png')
        plt.clf()

if __name__ == '__main__':
    # simple_stats('ogbn-arxiv', 1)
    cache_hit_plots()