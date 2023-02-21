import pandas as pd
import os
from typing import List

def load_df(model_name: str, path: str, graph_name: str, batch_size: int) -> pd.DataFrame: 
    """Load a Pandas DataFrame corresponding to a trace of inference latencies

    Args:
        model_name (str): GNN model, e.g. 'GCN', 'GAT', 'SAGE'
        path (str): Path where traces are located
        graph_name (str): Graph to be considered, e.g. 'ogbn-products'
        batch_size (int): Batch size of inference request

    Returns:
        pd.DataFrame: DataFrame containing inference latencies for specified model, graph, and batch size
    """
    return pd.read_csv(os.path.join(
                    path, model_name, f'{graph_name}-{batch_size}.csv'))

def load_dfs(model_name: str, path: str, graph_names: List[str], batch_sizes: List[int]) -> pd.DataFrame: 
    dfs = []
    for name in graph_names:
        for batch_size in batch_sizes:
            dfs.append(load_df(model_name, path, name, batch_size))
    return pd.concat(dfs)
