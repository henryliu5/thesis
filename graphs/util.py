import pandas as pd
import os
from typing import List
import glob

def load_df(model_name: str, path: str, graph_name: str, batch_size: int, trials: int = 3, agg='avg') -> pd.DataFrame: 
    """Load a Pandas DataFrame corresponding to a trace of inference latencies

    Args:
        model_name (str): GNN model, e.g. 'GCN', 'GAT', 'SAGE'
        path (str): Path where traces are located
        graph_name (str): Graph to be considered, e.g. 'ogbn-products'
        batch_size (int): Batch size of inference request
        trials (int): Number of trials to colelct
        agg (str): Which dataframe to report ()

    Returns:
        pd.DataFrame: DataFrame containing inference latencies for specified model, graph, and batch size
    """
    dfs = []
    best = None
    index = -1
    for trial in range(trials):
        trial_df = pd.read_csv(os.path.join(path, model_name, f'{graph_name}-{batch_size}-{trial}.csv'))
        dfs.append(trial_df)
        
        avg_exec = trial_df['total'].mean()
        if best == None or avg_exec < best:
            best = avg_exec
            index = trial

    return dfs[index]

def load_df_cache_info(model_name: str, path: str, graph_name: str, batch_size: int, trials: int = 3, agg='avg') -> pd.DataFrame: 
    """Load a Pandas DataFrame corresponding to a trace of inference latencies

    Args:
        model_name (str): GNN model, e.g. 'GCN', 'GAT', 'SAGE'
        path (str): Path where traces are located
        graph_name (str): Graph to be considered, e.g. 'ogbn-products'
        batch_size (int): Batch size of inference request
        trials (int): Number of trials to colelct
        agg (str): Which dataframe to report ()

    Returns:
        pd.DataFrame: DataFrame containing inference latencies for specified model, graph, and batch size
    """
    dfs = []
    best = None
    index = -1
    for trial in range(trials):
        trial_df = pd.read_csv(os.path.join(path, model_name, f'{graph_name}-{batch_size}-{trial}.csv'))
        dfs.append(trial_df)
        
        avg_exec = trial_df['total'].mean()
        if best == None or avg_exec < best:
            best = avg_exec
            index = trial
    print('using index', index)
    return pd.read_csv(os.path.join(path, model_name + "_cache_info", f'{graph_name}-{batch_size}-{index}.csv'))


def load_df_throughput(model_name: str, path: str, graph_name: str, batch_size: int) -> pd.DataFrame: 
    """Load a Pandas DataFrame corresponding to a trace of inference latencies

    Args:
        model_name (str): GNN model, e.g. 'GCN', 'GAT', 'SAGE'
        path (str): Path where traces are located
        graph_name (str): Graph to be considered, e.g. 'ogbn-products'
        batch_size (int): Batch size of inference request
        trials (int): Number of trials to colelct
        agg (str): Which dataframe to report ()

    Returns:
        pd.DataFrame: DataFrame containing inference latencies for specified model, graph, and batch size
    """
    dfs = []

    glob_match_path = os.path.join(path, model_name + "_throughput", f'{graph_name}-{batch_size}-*-*-*.csv')
    files = glob.glob(glob_match_path)
    
    for file in files:
        dfs.append(pd.read_csv(file))

    df = pd.concat(dfs, ignore_index=True)

    DROP_NUM = 1

    num_trials = df['trial'].max()

    # Drop lowest and highest
    df = df.groupby(['num_devices', 'executors_per_store'])['throughput (req/s)'].nlargest(num_trials - 1).reset_index(level=2, drop=True)
    df = df.reset_index()
    df = df.groupby(['num_devices', 'executors_per_store'])['throughput (req/s)'].nsmallest(num_trials - 2).reset_index(level=2, drop=True)
    return df.to_frame()

def load_df_throughput_p99_latency(model_name: str, path: str, num_stores: int, executors_per_store: int, percentile: int = 0.99) -> pd.DataFrame: 
    """Load a Pandas DataFrame corresponding to a trace of inference latencies

    Args:
        model_name (str): GNN model, e.g. 'GCN', 'GAT', 'SAGE'
        path (str): Path where traces are located
        graph_name (str): Graph to be considered, e.g. 'ogbn-products'
        batch_size (int): Batch size of inference request
        trials (int): Number of trials to colelct
        agg (str): Which dataframe to report ()

    Returns:
        pd.DataFrame: DataFrame containing inference latencies for specified model, graph, and batch size
    """
    dfs = []

    glob_match_path = os.path.join(path, model_name + "_breakdown_with_trials", f'*-*-*-*-*-*.csv')
    files = glob.glob(glob_match_path)
    for file in files:
        dfs.append(pd.read_csv(file))

    df = pd.concat(dfs, ignore_index=True)
    DROP_NUM = 1

    df = df[df['num_stores'] == num_stores]
    df = df[df['executors_per_store'] == executors_per_store]
    num_trials = df['trial'].max() + 1

    # Drop lowest and highest
    print(df)
    df = df.groupby(['trial']).quantile(percentile, numeric_only=True)['exec_time_since_generated']
    df = df.nlargest(num_trials - 1).reset_index(drop=True)
    # df = df.nsmallest(num_trials - 2).reset_index(drop=True)
    df = df.nsmallest(1).reset_index(drop=True)
    df = df.to_frame()
    df[f'P99 Latency'] = df['exec_time_since_generated']
    return df

def load_dfs(model_name: str, path: str, graph_names: List[str], batch_sizes: List[int]) -> pd.DataFrame: 
    dfs = []
    for name in graph_names:
        for batch_size in batch_sizes:
            dfs.append(load_df(model_name, path, name, batch_size, trials=3))
    return pd.concat(dfs)
