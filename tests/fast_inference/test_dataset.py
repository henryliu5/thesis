from fast_inference.dataset import InferenceDataset
import torch as th
import pytest

@pytest.mark.skip(reason="slow test, run manually")
def test_inference_dataset_reload():
    """ Check reloading dataset from disk works correctly """
    graph1 = InferenceDataset('reddit', 0.1, force_reload=True, verbose=True)
    assert(hasattr(graph1, '_orig_graph'))
    # graph2 should be loaded from disk
    graph2 = InferenceDataset('reddit', 0.1, verbose=True)
    assert(not hasattr(graph2, '_orig_graph'))
    assert(graph1._num_infer_targets == graph2._num_infer_targets)

    # Dataset 1 should generate the trace
    trace1 = graph1.create_inference_trace()
    # Dataset 2 shoud just load from disk
    trace2 = graph2.create_inference_trace()
    
    assert(th.all(th.eq(trace1.nids, trace2.nids)))
    assert(th.all(th.eq(trace1.features, trace2.features)))

    # Try generating a small trace
    # In this case, the Reddit dataset and inference parition info should all be on disk
    # This means that the only work that should be done is just picking random indices
    trace3 = graph2.create_inference_trace(100)
    # TODO test edges are equal, need to go through list of dictionaries

def test_inference_dataset_partition():
    infer_data = InferenceDataset('reddit', 0.9, verbose=True)
    pruned_g = infer_data[0]
    trace = infer_data.create_inference_trace()
    print(pruned_g)
    print(pruned_g.num_nodes())
    print(infer_data._num_infer_targets)

    # 1. Inference targets will be "present" in the pruned graph to preserve node ids
    assert(th.all(pruned_g.has_nodes(trace.nids)))

    # However, there will be no such edges
    assert(len(pruned_g.in_edges(trace.nids, 'eid')) == 0)
    assert(len(pruned_g.out_edges(trace.nids, 'eid')) == 0)
    
    # 2. Inference edges should connect to nodes present in the pruned graph
    in_nids = trace.edges.in_edge_endpoints
    out_nids = trace.edges.out_edge_endpoints
    assert(th.all(pruned_g.has_nodes(in_nids)))
    assert(th.all(pruned_g.has_nodes(out_nids)))

    # 3. Inference edges should not connect to other inference targets
    s = set(trace.nids.tolist())
    assert(all(x not in s for x in in_nids))
    assert(all(x not in s for x in out_nids))

def test_inference_dataset_subgraph():
    infer_data = InferenceDataset('reddit', 0.1, verbose=True)
    infer_data.create_inference_trace(trace_len=256000, subgraph_bias=0.8)

if __name__ == '__main__':
    test_inference_dataset_subgraph()