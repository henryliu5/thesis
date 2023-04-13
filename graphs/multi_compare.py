import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import glob
plt.style.use('seaborn')

cache_types = ['cpp', 'cpp_lock']

dfs = []    
for c in cache_types:
    files = glob.glob(f'throughput_testing/gpu/pinned/uniform/{c}_0.2/GCN_breakdown/{c}-*-*-*-*.csv')
    for file in files:
        dfs.append(pd.read_csv(file))

df = pd.concat(dfs, ignore_index=True)
print(df.columns)

df = df[df.executors_per_store == 6]
df1 = df[df.cache_type == 'cpp']
df2 = df[df.cache_type == 'cpp_lock']

df1 = df1.mean()
df2 = df2.mean()

pd.options.display.float_format = "{:,.5f}".format
print(df1 - df2)

# df = df[df.executors_per_store != 16]
# df['acquire peer lock'] *= 1000
g = sns.displot(data=df, x='acquire peer lock', kind="ecdf", col='num_stores', row='cache_type', hue='executors_per_store')
# plt.xlim(0, 5)
plt.tight_layout()
plt.savefig(f'Lock_Throughput_conflicts.png', bbox_inches='tight', dpi=250)
plt.clf()
print(df)