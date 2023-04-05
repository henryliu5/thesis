import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import os

def plot_latency_breakdown(model_name, graph_names, path, title = ""):
    batch_sizes = [256]

    dfs = []
    for name in graph_names:
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
    # NOTE comment in below line for percetanges, will prob want to update label too
    # avg_df[['sampling', 'CPU-GPU copy', 'model']] = avg_df[['sampling',
                                                                #   'CPU-GPU copy', 'model']].div(avg_df.total, axis=0).mul(100)
    melted = avg_df.melt(id_vars=['name', 'batch_size'],
                         value_vars=['sampling',
                                     'CPU-GPU copy', 'feature gather', 'model', 'total'],
                         var_name="type",
                         value_name="percentage", ignore_index=True)

    # Dict for plot axis - this way all plots have the same scale
    axis_height = {'reddit': 2.5, 'cora': 0.08, 'ogbn-products': 0.2, 'ogbn-papers100M': 0.030}

    fig, axs = plt.subplots(2, 2)
    width = 0.35
    for i, name in enumerate(graph_names):
        ax = axs[i // 2][i % 2]
        ax.set_ylim(0, axis_height[name])
        batch_sizes = ['1', '64', '128', '256']
        local_df = melted.loc[melted['name'] == name]
        # for b in batch_sizes:
        # batch_df = local_df.loc[local_df['batch_size'] == b]
        batch_df = local_df
        model = batch_df.loc[batch_df['type']
                             == 'model']['percentage'].to_numpy()
        cg_copy = batch_df.loc[batch_df['type'] ==
                               'CPU-GPU copy']['percentage'].to_numpy()
        feat_gather = batch_df.loc[batch_df['type'] ==
                               'feature gather']['percentage'].to_numpy()
        sampling = batch_df.loc[batch_df['type'] ==
                                'sampling']['percentage'].to_numpy()
        total = batch_df.loc[batch_df['type'] ==
                        'total']['percentage'].to_numpy()

        # Add total with label
        # total_bar = ax.bar(batch_sizes, total, width, label='total')
        

        ax.bar(batch_sizes, model, width, label='model')
        ax.bar(batch_sizes, cg_copy, width, label='CPU-GPU copy', bottom=model)
        ax.bar(batch_sizes, feat_gather, width,
               label='feature gather', bottom=np.add(model, cg_copy))
        sampling_bar = ax.bar(batch_sizes, sampling, width,
               label='sampling', bottom=model + cg_copy + feat_gather)

        ax.bar_label(sampling_bar, labels=[f'{(x):.3f}' for x in total])

        ax.set_ylabel('time (s)')
        ax.set_xlabel('batch size')
        ax.set_title(name)

    # Get just the three labels - not 3 x 4
    handles, labels = ax.get_legend_handles_labels()
    # Place legend in right spot (bottom)
    fig.legend(handles[::-1], labels[::-1], bbox_to_anchor=(0.85, 0),  ncol=4)
    fig.tight_layout()
    # Shift plots down
    fig.subplots_adjust(top=0.85)
    # Set big figure title
    fig.suptitle(f'{model_name} inference latency {title}', fontsize=16)
    # Save
    plt.savefig(f'{model_name}_latency_breakdown.png',
                bbox_inches='tight', dpi=250)


if __name__ == '__main__':
    # names = ['reddit', 'cora', 'ogbn-products', 'ogbn-papers100M']
    # path = 'benchmark/data/new_baseline'
    # title = ""

    names = ['ogbn-products']
    path = 'benchmark/data/new_cache_gpu'
    title = "(GPU sampling, 20% static cache)"

    plot_latency_breakdown('GCN', names, path, title)
    # plot_latency_breakdown('SAGE', names, path, title)
    # plot_latency_breakdown('GAT', names, path, title)
