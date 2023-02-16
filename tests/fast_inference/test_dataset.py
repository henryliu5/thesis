from fast_inference.dataset import InferenceDataset
import torch as th

def test_inference_dataset_reload():
    """ Check reloading dataset from disk works correctly """
    graph1 = InferenceDataset('reddit', 0.1, force_reload=True, verbose=True)
    # graph2 should be loaded from disk
    graph2 = InferenceDataset('reddit', 0.1, verbose=True)

    assert(graph1._num_infer_targets == graph2._num_infer_targets)
    assert(th.all(th.eq(graph1.trace_nids, graph2.trace_nids)))
    assert(th.all(th.eq(graph1.trace_features, graph2.trace_features)))
    # TODO test edges are equal, need to go through list of dictionaries

def test_inference_dataset_partition():
    infer_data = InferenceDataset('reddit', 1, verbose=True)
    pruned_g = infer_data[0]
    print(pruned_g)
    print(pruned_g.num_nodes())
    print(infer_data._num_infer_targets)

    # 1. Inference targets will be "present" in the pruned graph to preserve node ids
    assert(th.all(pruned_g.has_nodes(infer_data.trace_nids)))

    # However, there will be no such edges
    assert(len(pruned_g.in_edges(infer_data.trace_nids, 'eid')) == 0)
    assert(len(pruned_g.out_edges(infer_data.trace_nids, 'eid')) == 0)
    
    # 2. Inference edges should connect to nodes present in the pruned graph
    in_nids = th.cat([x["in"] for x in infer_data.trace_edges])
    out_nids = th.cat([x["out"] for x in infer_data.trace_edges])
    assert(th.all(pruned_g.has_nodes(in_nids)))
    assert(th.all(pruned_g.has_nodes(out_nids)))

    # 3. Inference edges should not connect to other inference targets
    s = set(infer_data.trace_nids.tolist())
    assert(all(x not in s for x in in_nids))
    assert(all(x not in s for x in out_nids))
