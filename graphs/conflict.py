import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import glob
plt.style.use('seaborn')

def timing():
    cache_type = 'cpp_lock'

    files = glob.glob(f'pipeline_dataload_time/{cache_type}-*-*-*-*.csv')
    dfs = []    
    for file in files:
        dfs.append(pd.read_csv(file))

    
    df = pd.concat(dfs, ignore_index=True)
    # df = df[df.executors_per_store != 16]

    df['feature gather (ms)'] = 1000 * df['feature gather']
    df['num GPUs'] = df['num_stores']
    df['InferenceEngines per GPU'] = df['executors_per_store']
    g = sns.displot(data=df, x='feature gather (ms)', kind="ecdf", col='num GPUs', hue='InferenceEngines per GPU', col_wrap=2)
    # plt.xlim(0, 5)
    plt.tight_layout()
    plt.savefig(f'Lock_Timing.png', bbox_inches='tight', dpi=250)
    plt.clf()
    print(df)

def contention():
    cache_type = 'cpp_lock'

    files = glob.glob(f'pipeline_conflicts/{cache_type}-*-*-*-*.csv')
    dfs = []    
    for file in files:
        dfs.append(pd.read_csv(file))

    
    df = pd.concat(dfs, ignore_index=True)
    # df = df[df.executors_per_store != 16]

    df['wait_time (ms)'] = 1000 * df['wait_time']
    df['num GPUs'] = df['num_stores']
    df['InferenceEngines per GPU'] = df['executors_per_store']
    g = sns.displot(data=df, x='wait_time (ms)', kind="ecdf", col='num GPUs', hue='InferenceEngines per GPU', col_wrap=2, log_scale=True)
    # plt.xlim(0, 5)
    plt.suptitle(
        f'Time spent waiting on lock acquisition (ms)')
    plt.tight_layout()
    plt.savefig(f'Lock_Conflicts.png', bbox_inches='tight', dpi=250)
    plt.clf()
    print(df)

if __name__ == '__main__':
    contention()
    timing()