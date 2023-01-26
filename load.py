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

if __name__ == '__main__':
    simple_stats('ogbn-products', 1)