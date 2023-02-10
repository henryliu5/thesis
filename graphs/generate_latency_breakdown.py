import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_latency_breakdown(model_name, path):
    names = ['reddit', 'cora', 'ogbn-products', 'ogbn-papers100M']
    batch_sizes = [1, 64, 128, 256]

    dfs = []
    for name in names:
        for batch_size in batch_sizes:
            dfs.append(pd.read_csv(os.path.join(
                path, model_name, f'{name}-{batch_size}.csv')))

    df = pd.concat(dfs)
    # print(df)

    g = sns.catplot(df, x='batch_size', y='total',
                    col='name', kind="bar", col_wrap=2)

    # iterate through axes
    for ax in g.axes.ravel():
        # add annotations
        for c in ax.containers:
            labels = [f'{(v.get_height()):.3f}' for v in c]
            ax.bar_label(c, labels=labels, padding=8)
        ax.margins(y=0.2)

    plt.savefig(f'{model_name}_latency_totals.png')
    plt.clf()

    # Compute averages
    avg_df = df.groupby(['name', 'batch_size'], as_index=False).mean()
    print(avg_df)
    # Compute percentages of each type
    avg_df[['(cpu) sampling', 'CPU-GPU copy', 'model']] = avg_df[['(cpu) sampling',
                                                                  'CPU-GPU copy', 'model']].div(avg_df.total, axis=0).mul(100)
    melted = avg_df.melt(id_vars=['name', 'batch_size'],
                         value_vars=['(cpu) sampling',
                                     'CPU-GPU copy', 'model'],
                         var_name="type",
                         value_name="percentage", ignore_index=True)

    fig, axs = plt.subplots(2, 2)
    width = 0.35
    for i, name in enumerate(names):
        ax = axs[i // 2][i % 2]
        batch_sizes = ['1', '64', '128', '256']
        local_df = melted.loc[melted['name'] == name]
        # for b in batch_sizes:
        # batch_df = local_df.loc[local_df['batch_size'] == b]
        batch_df = local_df
        model = batch_df.loc[batch_df['type']
                             == 'model']['percentage'].to_numpy()
        cg_copy = batch_df.loc[batch_df['type'] ==
                               'CPU-GPU copy']['percentage'].to_numpy()
        sampling = batch_df.loc[batch_df['type'] ==
                                '(cpu) sampling']['percentage'].to_numpy()

        ax.bar(batch_sizes, model, width, label='model')
        ax.bar(batch_sizes, cg_copy, width, label='CPU-GPU copy', bottom=model)
        ax.bar(batch_sizes, sampling, width,
               label='(cpu) sampling', bottom=np.add(model, cg_copy))

        ax.set_ylabel('% req. time')
        ax.set_xlabel('batch size')
        ax.set_title(name)

    # Get just the three labels - not 3 x 4
    handles, labels = ax.get_legend_handles_labels()
    # Place legend in right spot (bottom)
    fig.legend(handles, labels, bbox_to_anchor=(0.87, 0),  ncol=3)
    fig.tight_layout()
    # Shift plots down
    fig.subplots_adjust(top=0.85)
    # Set big figure title
    fig.suptitle(f'{model_name} inference latency', fontsize=16)
    # Save
    plt.savefig(f'{model_name}_latency_breakdown.png',
                bbox_inches='tight', dpi=100)


if __name__ == '__main__':
    path = 'benchmark/data/timing_breakdown'
    plot_latency_breakdown('GCN', path)
    # plot_latency_breakdown('SAGE')
    # plot_latency_breakdown('GAT')
