import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import glob
plt.style.use('seaborn')

cache_types = [
                # 'static', 
               'count', 
               'cpp',
                 'cpp_lock']

dfs = []    
for c in cache_types:
    files = glob.glob(f'multiple_throughput/gpu/pinned/uniform/{c}_0.2/GCN_breakdown/{c}-*-*-*-*.csv')
    for file in files:
        dfs.append(pd.read_csv(file))

df = pd.concat(dfs, ignore_index=True)
print(df.columns)

type1 = 'cpp'
type2 = 'cpp_lock'
df = df[df.executors_per_store == 8]
df = df[df.num_stores == 2]

df1 = df[df.cache_type == type1]
df2 = df[df.cache_type == type2]

df1 = df1.mean(numeric_only=True)
df2 = df2.mean(numeric_only=True)

pd.options.display.float_format = "{:,.5f}".format
print(df1 - df2)

print(type1, 'p99 latency', df[df.cache_type == type1].quantile(0.99, numeric_only=True)['exec request'] * 1000, 'ms')
print(type2, 'p99 latency', df[df.cache_type == type2].quantile(0.99, numeric_only=True)['exec request'] * 1000, 'ms')

print(type1, 'estimated throughput', df1['num_stores'] * df1['executors_per_store'] / df1['exec request'])
print(type2, 'estimated throughput', df2['num_stores'] * df2['executors_per_store'] / df2['exec request'])
# df = df[df.executors_per_store != 16]
df['acquire peer lock'] *= 1000
g = sns.displot(data=df, x='acquire peer lock', kind="ecdf", col='executors_per_store', hue='cache_type')
plt.xlim(0, 15)
plt.tight_layout()
plt.savefig(f'Throughput_conflicts.png', bbox_inches='tight', dpi=250)
plt.clf()
print(df)