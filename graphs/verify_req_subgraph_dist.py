''' Utiliy to check the cyclical nature of request traces when enabled'''
from fast_inference.dataset import InferenceDataset
from util import load_df
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import math
from tqdm import tqdm

def main():
    infer_data = InferenceDataset('ogbn-products', 0.1, verbose=True)
    bias = 0.8
    trace = infer_data.create_inference_trace(trace_len=256000, subgraph_bias=bias)

    requests_per_partition = int(math.ceil(len(trace) / infer_data._num_partitions))
    part_mapping = infer_data._orig_nid_partitions

    for partition in range(infer_data._num_partitions):
        total = 0
        in_subgraph = 0
        for i in range(requests_per_partition):
            cur_index = requests_per_partition * partition + i
            if cur_index >= len(trace):
                break
            if part_mapping[trace.nids[cur_index]] == partition:
                in_subgraph += 1
            total += 1
        print(f'Partition {partition}, {in_subgraph/total * 100} % in subgraph, bias is {bias}')

if __name__ == '__main__':
    main()