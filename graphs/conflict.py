import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import glob
plt.style.use('seaborn')

if __name__ == '__main__':
    cache_type = 'cpp_lock'

    files = glob.glob(f'pipeline_conflicts/{cache_type}-*-*-*-*.csv')
    dfs = []    
    for file in files:
        dfs.append(pd.read_csv(file))

    
    df = pd.concat(dfs, ignore_index=True)
    # df = df[df.executors_per_store != 16]

    df['wait_time (ms)'] = 1000 * df['wait_time']
    g = sns.displot(data=df, x='wait_time (ms)', kind="ecdf", col='num_stores', hue='executors_per_store', col_wrap=2, log_scale=True)
    # plt.xlim(0, 5)
    plt.tight_layout()
    plt.savefig(f'Lock_Conflicts.png', bbox_inches='tight', dpi=250)
    plt.clf()
    print(df)

    # cache_type = 'cpp'

    # files = glob.glob(f'throughput_testing/gpu/pinned/uniform/{cache_type}_0.2/GCN_breakdown/{cache_type}-*-*-*-*.csv')
    # dfs = []    
    # for file in files:
    #     dfs.append(pd.read_csv(file))

    # df = pd.concat(dfs, ignore_index=True)
    # print(df)
    # # df = df[df.executors_per_store != 16]
    # # df['acquire peer lock'] *= 1000
    # g = sns.displot(data=df, x='dataloading', kind="ecdf", col='num_stores', hue='executors_per_store', col_wrap=2)
    # # plt.xlim(0, 5)
    # plt.tight_layout()
    # plt.savefig(f'Lock_Throughput_conflicts.png', bbox_inches='tight', dpi=250)
    # plt.clf()
    # print(df)