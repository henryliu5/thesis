from ogb.nodeproppred import DglNodePropPredDataset
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
from dataclasses import dataclass
import random


@dataclass(frozen=True)
class InferenceTrace:
    """ Tensor of node IDs representing to inference targets in the trace"""
    nids: th.Tensor
    """ Tensor of features for each index in the trace"""
    features: th.Tensor
    """ Returns a list of dicts, each dict containing the in and out edges for a node in the trace.
        Keys are 'in' and 'out', values are tensors of node IDs.
    """
    edges: List[Dict[str, th.Tensor]]

    def __post_init__(self):
        # Must all have same dimension
        assert (self.nids.shape[0] == self.features.shape[0])
        assert (self.features.shape[0] == len(self.edges))

    def __len__(self):
        return self.nids.shape[0]


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
        th.manual_seed(rand_seed)
        random.seed(rand_seed)
        np.random.seed(rand_seed)
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
        makedirs(self.raw_path)
        self.download()

    def download(self):
        if self._orig_name.startswith('ogbn'):
            # if self._orig_name == 'ogbn-products-METIS':
            #     dataset = DglNodePropPredDataset('ogbn-products')
            #     graph, _ = dataset[0]
            #     # Partition 5 ways, use first one as the new graph
            #     graph = dgl.metis_partition(graph, 5)[0]
            # else:
            self._dataset = DglNodePropPredDataset(self._orig_name)
            self._orig_graph, _ = self._dataset[0]
            if self._orig_name == 'ogbn-arxiv':
                # Add reverse edges since ogbn-arxiv is unidirectional.
                self._orig_graph = dgl.add_reverse_edges(self._orig_graph)
            self._num_classes = self._dataset.num_classes
            return

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
        self._num_classes = self._dataset.num_classes

    def process(self):
        """Split graph into training and inference nodes"""

        # TODO: Broken if load() happens to crash with force_reload off since this will be called but _download() will just return, so orig_graph won't exist
        self._create_inference_partition()

        self.graphs = [self._pruned_graph]

    def _create_inference_partition(self):
        self._num_infer_targets = int(
            self._orig_graph.number_of_nodes() * self._target_percent)
        self._infer_target_nids = th.randperm(self._orig_graph.number_of_nodes())[
            :self._num_infer_targets]

        infer_target_mask = th.zeros(
            self._orig_graph.number_of_nodes(), dtype=th.bool)
        infer_target_mask[self._infer_target_nids] = True
        print(
            f'Original graph nodes: {self._orig_graph.number_of_nodes()}, Inference target percent: {self._target_percent}, num_infer_targets: {self._num_infer_targets}')

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

        self._get_inference_target_info()

    def _get_inference_target_info(self):
        """ Generate and store information related to inference targets on disk or
            load it from disk. This includes features, nids, and edges for all inference targets.
        """
        infer_path = os.path.join(self.save_path, f'infer_info.pkl')

        if os.path.exists(infer_path):
            infer_info = load_info(infer_path)
            self.infer_nids = infer_info['infer_nids']
            self.infer_features = infer_info['infer_features']
            self.infer_edges = infer_info['infer_edges']
        else:
            assert (hasattr(self, '_infer_target_nids')
                    ), "Illegal state when creating inference trace, is the inference_dataset directory up to date?"
            self.infer_nids = self._infer_target_nids
            self.infer_features = self._orig_graph.ndata['feat'][self._infer_target_nids]
            print('Generating edges for ALL inference targets:')
            self.infer_edges = []
            for i in tqdm(range(len(self.infer_nids))):
                in_neighbors = self._orig_graph.in_edges(
                    self.infer_nids[i], 'uv')[0]
                out_neighbors = self._orig_graph.out_edges(
                    self.infer_nids[i], 'uv')[1]
                self.infer_edges.append(
                    {'in': in_neighbors, 'out': out_neighbors})

            save_info(infer_path, {'infer_nids': self.infer_nids,
                                   'infer_features': self.infer_features,
                                   'infer_edges': self.infer_edges})

    def create_inference_trace(self, trace_len: int = 256_000) -> InferenceTrace:
        """Create a trace of inference requests

        Since self._orig_graph has had edges connecting inference targets removed,
        we can generate valid requests just by randomly selecting inference targets.

        Args:
            trace_len (int): Maximum length of trace to be generated
        """
        # Load entire trace from disk if available
        # Will be stored under path for dataset, file name denotes length
        trace_path = os.path.join(self.save_path, f'trace_{trace_len}.pkl')
        if os.path.exists(trace_path):
            print('Loading trace from', trace_path)
            return load_info(trace_path)['trace']

        self._get_inference_target_info()
        assert (self.infer_nids.shape[0] == self.num_infer_targets)

        trace_len = min(trace_len, self.num_infer_targets)
        print(f'Generating inference trace of length {trace_len}...')
        generated_indices = th.randperm(self.num_infer_targets)[:trace_len]

        trace_nids = self.infer_nids[generated_indices]
        trace_features = self.infer_features[generated_indices]

        print('Generating edges:')
        trace_edges = []
        for idx in tqdm(generated_indices):
            trace_edges.append(self.infer_edges[idx])

        trace = InferenceTrace(trace_nids, trace_features, trace_edges)
        # Cache on disk
        save_info(trace_path, {'trace': trace})
        return trace

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
        save_info(info_path, {'num_infer_targets': self._num_infer_targets,
                              'num_classes': self._num_classes})

    def load(self):
        # load processed data from directory `self.save_path`
        graph_path = os.path.join(self.save_path, '_dgl_graph.bin')
        self.graphs, _ = load_graphs(graph_path)
        self._pruned_graph = self.graphs[0]

        info_path = os.path.join(self.save_path, '_info.pkl')
        trace_info = load_info(info_path)
        self._num_infer_targets = trace_info['num_infer_targets']
        self._num_classes = trace_info['num_classes']

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_path, '_dgl_graph.bin')
        info_path = os.path.join(self.save_path, '_info.pkl')
        return os.path.exists(graph_path) and os.path.exists(info_path)

    def __repr__(self):
        return (
            f'Dataset("{self.name}", num_graphs={len(self)},'
            + f" save_path={self.save_path},"
            + f" num_infer_targets={self._num_infer_targets})"
        )

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_infer_targets(self):
        return self._num_infer_targets
