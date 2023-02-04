from dgl.data import DGLDataset, RedditDataset, CoraFullDataset, CiteseerGraphDataset
import os
import shutil
import dgl
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
import torch as th
import numpy as np
from tqdm import tqdm
from typing import List, Dict


class InferenceDataset(DGLDataset):
    """
    DGLGraph partitioned to create a split between "training" and "inference" nodes.
    An InferenceDataset includes a DGLGraph with a subset of nodes removed for use as "inference targets".
    Also included is a trace of inference requests using the inference targets.

    This class ALL nodes in the graph when creating the inference set.

    Parameters
    ----------
    url : str
        URL to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """

    TRACE_LEN = 10000

    def __init__(self,
                 name,
                 target_percent,
                 rand_seed=143253,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False,
                 **kwargs):
        self._orig_name = name
        self._internal_kwargs = kwargs
        self._target_percent = target_percent
        self._rand_seed = rand_seed
        self._verbose = verbose

        super(InferenceDataset, self).__init__(name=f'inference_{name}_{target_percent}',
                                               url=None,
                                               raw_dir=raw_dir,
                                               save_dir=save_dir,
                                               force_reload=force_reload,
                                               verbose=verbose)

    def _download(self):
        """Download dataset by calling ``self.download()``
        if the dataset does not exists under ``self.raw_path``.

        By default ``self.raw_path = os.path.join(self.raw_dir, self.name)``
        One can overwrite ``raw_path()`` function to change the path.
        """
        # HACK Make force reload delete the inference directory
        if os.path.exists(self.raw_path) and self._force_reload:
            print(f'Force reload is set, deleting {self.raw_path}')
            shutil.rmtree(self.raw_path)

        if os.path.exists(self.raw_path):
            return

        makedirs(self.raw_dir)
        self.download()

    def download(self):
        if self._orig_name == 'reddit':
            self._dataset = RedditDataset(verbose=self._verbose,
                                          **self._internal_kwargs)
        elif self._orig_name == 'cora':
            self._dataset = CoraFullDataset(verbose=self._verbose,
                                            **self._internal_kwargs)
        elif self._orig_name == 'citeseer':
            self._dataset = CiteseerGraphDataset(verbose=self._verbose,
                                                 **self._internal_kwargs)

        self._orig_graph = self._dataset[0]

    def process(self):
        """Split graph into training and inference nodes"""

        # TODO: Broken if load() happens to crash with force_reload off since this will be called but _download() will just return, so orig_graph won't exist
        self.create_inference_partition()
        self.create_inference_trace(self.TRACE_LEN)

        self.graphs = [self._pruned_graph]

    def create_inference_partition(self):

        infer_target_mask = th.BoolTensor(np.random.choice([False, True],
                                                           size=(
                                                               self._orig_graph.number_of_nodes(),),
                                                           p=[1 - self._target_percent, self._target_percent]))
        num_infer_targets = infer_target_mask.sum()
        print(
            f'Original graph nodes: {self._orig_graph.number_of_nodes()}, Inference target percent: {self._target_percent}, num_infer_targets: {num_infer_targets}')

        id_arr = th.arange(self._orig_graph.number_of_nodes())
        self._infer_target_nids = th.masked_select(id_arr, infer_target_mask)
        self._num_infer_targets = infer_target_mask.sum()

        # 1. Remove edges between nodes that are inference targets
        #    This ensures that inference requests are independent

        self._orig_graph.ndata['is_infer_target'] = infer_target_mask
        self._orig_graph.apply_edges(lambda edges: {'should_remove': th.logical_and(
            edges.dst['is_infer_target'], edges.src['is_infer_target'])})
        eids_to_remove = self._orig_graph.filter_edges(
            lambda edges: edges.data['should_remove'])

        # NOTE: _orig_graph is the original graph with inference target -> interence target edges removed
        self._orig_graph.remove_edges(eids_to_remove)

        # 2. Remove edges between inference targets and non-inference targets
        #    This creates the graph partition

        # self._pruned_graph = dgl.node_subgraph(self._orig_graph, th.masked_select(id_arr, th.logical_not(infer_target_mask)), relabel_nodes=False)
        self._orig_graph.ndata['is_not_infer_target'] = th.logical_not(
            infer_target_mask)
        self._orig_graph.apply_edges(lambda edges: {'should_keep': th.logical_and(
            edges.dst['is_not_infer_target'], edges.src['is_not_infer_target'])})
        eids_to_keep = self._orig_graph.filter_edges(
            lambda edges: edges.data['should_keep'])

        # clean up "features"
        self._orig_graph.ndata.pop('is_infer_target')
        self._orig_graph.edata.pop('should_remove')
        self._orig_graph.ndata.pop('is_not_infer_target')
        self._orig_graph.edata.pop('should_keep')

        self._pruned_graph = dgl.edge_subgraph(
            self._orig_graph, eids_to_keep, relabel_nodes=False)

    def create_inference_trace(self, trace_len: int):
        """Create a trace of inference requests

        Since self._orig_graph has had edges connecting inference targets removed,
        we can generate valid requests just by randomly selecting inference targets.

        Args:
            trace_len (int): Length of trace to be generated
        """
        print('Generating inference trace...')
        self._trace_nids = th.tensor(np.random.choice(
            self._infer_target_nids, size=(trace_len,), replace=True))
        self._trace_features = self._orig_graph.ndata['feat'][self._trace_nids]
        print('Generating edges:')
        self._trace_edges = []
        for i in tqdm(range(trace_len)):
            in_neighbors = self._orig_graph.in_edges(
                self._trace_nids[i], 'uv')[0]
            out_neighbors = self._orig_graph.out_edges(
                self._trace_nids[i], 'uv')[1]
            self._trace_edges.append(
                {'in': in_neighbors, 'out': out_neighbors})

        assert (len(self._trace_nids) == len(self._trace_features))
        assert (len(self._trace_nids) == len(self._trace_edges))

    def __getitem__(self, idx):
        # get one example by index
        assert idx == 0, "This dataset has only one graph"
        return self._pruned_graph

    def __len__(self):
        # Node classification only, only 1 graph
        # TODO add other GNN tasks
        return 1

    def save(self):
        # save graphs and labels
        graph_path = os.path.join(self.save_path, '_dgl_graph.bin')
        save_graphs(graph_path, self.graphs)

        # save other information in python dict
        info_path = os.path.join(self.save_path, '_info.pkl')
        save_info(info_path, {'trace_nids': self._trace_nids,
                              'trace_features': self._trace_features,
                              'trace_edges': self._trace_edges,
                              'num_infer_targets': self._num_infer_targets})

    def load(self):
        # load processed data from directory `self.save_path`
        graph_path = os.path.join(self.save_path, '_dgl_graph.bin')
        self.graphs, _ = load_graphs(graph_path)
        self._pruned_graph = self.graphs[0]

        info_path = os.path.join(self.save_path, '_info.pkl')
        trace_info = load_info(info_path)
        self._trace_nids = trace_info['trace_nids']
        self._trace_features = trace_info['trace_features']
        self._trace_edges = trace_info['trace_edges']
        self._num_infer_targets = trace_info['num_infer_targets']

        assert (len(self._trace_nids) == len(self._trace_features))
        assert (len(self._trace_nids) == len(self._trace_edges))

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_path, '_dgl_graph.bin')
        info_path = os.path.join(self.save_path, '_info.pkl')
        return os.path.exists(graph_path) and os.path.exists(info_path)

    def __repr__(self):
        return (
            f'Dataset("{self.name}", num_graphs={len(self)},'
            + f" save_path={self.save_path},"
            + f" trace_len={len(self._trace_nids)},"
            + f" num_infer_targets={self._num_infer_targets})"
        )

    @property
    def trace_nids(self) -> th.Tensor:
        """ Tensor of node IDs representing to inference targets in the trace"""
        return self._trace_nids

    @property
    def trace_features(self) -> th.Tensor:
        """ Tensor of features for each index in the trace"""
        return self._trace_features

    @property
    def trace_edges(self) -> List[Dict[str, th.Tensor]]:
        """ Returns a list of dicts, each dict containing the in and out edges for a node in the trace.
            Keys are 'in' and 'out', values are tensors of node IDs.
        """
        return self._trace_edges

    @property
    def num_infer_targets(self):
        return self._num_infer_targets
