import time
import dgl
import tqdm
import sys
import blosc2
import pandas as pd
import seaborn as sns
from gen import get_graph
import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

def try_compression(feature_data, element_size, feature_bytes, df_rows, name, batch_size):
    # codecs = [blosc2.Codec.BLOSCLZ, blosc2.Codec.LZ4, blosc2.Codec.LZ4HC, blosc2.Codec.ZLIB, blosc2.Codec.ZSTD]
    codecs = [blosc2.Codec.LZ4, blosc2.Codec.ZSTD]
    # filters = [blosc2.Filter.NOFILTER, blosc2.Filter.SHUFFLE, blosc2.Filter.BITSHUFFLE, blosc2.Filter.DELTA, blosc2.Filter.TRUNC_PREC]
    filters = [blosc2.Filter.NOFILTER, blosc2.Filter.DELTA]
    print('element size', element_size)
    for codec in codecs:
        for filter in filters:
            start = time.time()
            compressed = blosc2.compress(feature_data, typesize=element_size, codec=codec, filter=filter)
            time_elapsed = time.time() - start
            print(codec, filter, 'feature size:', sizeof_fmt(feature_bytes), sizeof_fmt(len(compressed)), time_elapsed)
            res = {'graph': [name], 'ratio': [len(compressed) / feature_bytes], 'codec': [str(codec) + str(filter)], 'time': [time_elapsed], 'batch_size': [batch_size]}

            df_rows.append(pd.DataFrame(res))

def compress_samples():
    datasets = ['ogbn-products', 'reddit', 'citeseer', 'cora']
    batch_sizes = [1, 256, 1024]
    df_rows = []
    for name in datasets:
        for batch_size in batch_sizes:
            g, train_nids = get_graph(name)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

            dataloader = dgl.dataloading.DataLoader(
                g, train_nids, sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0)

            for trial in range(3):
                input_nodes, output_nodes, blocks = next(iter(dataloader))
                # print(input_nodes, output_nodes, blocks)
                print("To compute {} nodes' outputs, we need {} nodes' input features".format(len(output_nodes), len(input_nodes)))
                features = blocks[0].srcdata['feat']
                feature_bytes = features.element_size() * features.nelement()

                feature_data = features.numpy().data
                print('feature min:', features.min(), 'max:', features.max())
                try_compression(feature_data, features.element_size(), feature_bytes, df_rows, name, batch_size)
                
    df = pd.concat(df_rows)
    df.to_csv('compress_table.pkl')

if __name__ == '__main__':
    # compress_samples()

    df = pd.read_csv('compress_table.csv')
    g = sns.catplot(df, x='batch_size', y='ratio', col='graph', kind="bar", hue="codec", col_wrap=2)
    # iterate through axes
    for ax in g.axes.ravel():
        # add annotations
        for c in ax.containers:
            labels = [f'{(v.get_height()):.2f}' for v in c]
            ax.bar_label(c, labels=labels, padding=8)
        ax.margins(y=0.2)
    plt.savefig('compress.png')
    plt.show()
    plt.clf()
    g = sns.catplot(df, x='batch_size', y='time', col='graph', kind="bar", hue="codec", col_wrap=2)

    # iterate through axes
    for ax in g.axes.ravel():
        # add annotations
        for c in ax.containers:
            labels = [f'{(v.get_height()):.3f}' for v in c]
            ax.bar_label(c, labels=labels, padding=8)
        ax.margins(y=0.2)
        
    plt.savefig('compress_time.png')
    plt.show()
    
