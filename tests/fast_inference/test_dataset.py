from fast_inference.dataset import InferenceDataset
import torch as th

def test_inference_dataset_reload():
    """ Check reloading dataset from disk works correctly """
    graph1 = InferenceDataset('reddit', 0.1, force_reload=True, verbose=True)
    # graph2 should be loaded from disk
    graph2 = InferenceDataset('reddit', 0.1, verbose=True)

    assert(graph1.num_infer_targets == graph2.num_infer_targets)
    assert(th.all(th.eq(graph1.trace_nids, graph2.trace_nids)))
    assert(th.all(th.eq(graph1.trace_features, graph2.trace_features)))
    # TODO test edges are equal, need to go through list of dictionaries