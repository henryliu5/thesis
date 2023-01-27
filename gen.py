import dgl
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dgl.data import RedditDataset
import torch as th
from tqdm import tqdm
import pickle

def get_graph(name):
    if name == 'reddit':
        dataset = RedditDataset()
        graph = dataset[0]
        train_nids = th.nonzero(graph.ndata['train_mask']).squeeze()
    elif name.startswith('ogbn'):
        if name == 'ogbn-products-METIS':
            dataset = DglNodePropPredDataset('ogbn-products')
            graph, _ = dataset[0]
            # Partition 5 ways, use first one as the new graph
            graph = dgl.metis_partition(graph, 5)[0]
        else:
            dataset = DglNodePropPredDataset(name)
            graph, _ = dataset[0]
        # Add reverse edges since ogbn-arxiv is unidirectional.
        graph = dgl.add_reverse_edges(graph)

        idx_split = dataset.get_idx_split()
        train_nids = idx_split['train']
    else:
        print('graph', name, 'not supported')
        exit()

    return graph, train_nids

def do_plot():
    graph, train_nids = get_graph('reddit')
    device = 'cpu'      # change to 'cuda' for GPU

    node_features = graph.ndata['feat']

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    train_dataloader = dgl.dataloading.DataLoader(
        # The following arguments are specific to DGL's DataLoader.
        graph,              # The graph
        train_nids,         # The node IDs to iterate over in minibatches
        sampler,            # The neighbor sampler
        device=device,      # Put the sampled MFGs on CPU or GPU
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=256,    # Batch size
        shuffle=True,       # Whether to shuffle the nodes for every epoch
        drop_last=False,    # Whether to drop the last incomplete batch
        num_workers=1       # Number of sampler processes
    )

    sizes = []

    with train_dataloader.enable_cpu_affinity():
        for input_nodes, output_nodes, mfgs in train_dataloader:
            sizes.append(len(input_nodes))
            if(len(sizes) > 1000):
                break

    # drop last of batch since it will likely be small
    sizes.pop()
    
    var = np.var(sizes)
    print(len(sizes), 'variance: ', var)

    df = pd.DataFrame(sizes, columns=['mfg size'])
    sns.displot(df, kde=True)
    plt.show()

def save_sampling(graph_name, batch_size, sampler, n):
    graph, train_nids = get_graph(graph_name)
    device = 'cpu'      # change to 'cuda' for GPU

    # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    train_dataloader = dgl.dataloading.DataLoader(
        # The following arguments are specific to DGL's DataLoader.
        graph,              # The graph
        train_nids,         # The node IDs to iterate over in minibatches
        sampler,            # The neighbor sampler
        device=device,      # Put the sampled MFGs on CPU or GPU
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=batch_size,    # Batch size
        shuffle=True,       # Whether to shuffle the nodes for every epoch
        drop_last=True,    # Whether to drop the last incomplete batch
        num_workers=0       # Number of sampler processes
    )

    nodes = [[]]

    # with train_dataloader.enable_cpu_affinity():
    #     it = iter(train_dataloader)
    #     for i in tqdm(range(n)):
    #         input_nodes, output_nodes, mfgs = next(it)
    #         np.append(sizes, len(input_nodes)) # sizes.append(len(input_nodes))
    #         if(len(sizes) >= n):
    #             break

    it = iter(train_dataloader)
    for i in tqdm(range(n)):
        input_nodes, output_nodes, mfgs = next(it)

        nodes.append(input_nodes)
        if(len(nodes) >= n):
            break

    filename = f'{graph_name}_{batch_size}'
    outfile = open(filename, 'wb')
    pickle.dump(nodes, outfile)
    outfile.close()

    # np.savez_compressed(f'{graph_name}_{batch_size}', nodes)

    # var = np.var(sizes)
    # print(len(sizes), 'variance: ', var)

    # df = pd.DataFrame(sizes, columns=['mfg size'])
    # sns.displot(df, kde=True)
    # plt.show()

def save_sort(name):
    dgl_g, _ = get_graph(name)
    out_degrees = dgl_g.out_degrees()
    sort_nid = th.argsort(out_degrees, descending=True).numpy()
    np.savez_compressed(f'{name}_ordered', sort_nid)

if __name__ == '__main__':
    # do_plot()
    batchs = [1]#, 8, 64, 128, 256, 512, 1024]
    for b in batchs:
        # save_sampling('reddit', b, dgl.dataloading.MultiLayerFullNeighborSampler(2), 10000)
        # save_sampling('ogbn-products', b, dgl.dataloading.MultiLayerFullNeighborSampler(2), 10000)
        # save_sampling('ogbn-arxiv', b, dgl.dataloading.MultiLayerFullNeighborSampler(2), 10000)
        save_sampling('ogbn-products-METIS', b, dgl.dataloading.MultiLayerFullNeighborSampler(2), 10000)
        # save_sort('reddit')
        # save_sort('ogbn-products')
        # save_sort('ogbn-arxiv')
        # save_sort('ogbn-products-METIS')