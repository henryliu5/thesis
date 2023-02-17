from fast_inference.dataset import InferenceDataset
from fast_inference.models.factory import load_model
from fast_inference.timer import enable_timers, Timer, print_timer_info, export_timer_info, clear_timers
import dgl
import torch
from tqdm import tqdm
import gc

def tracefunc(frame, event, arg, indent=[0]):
      if event == "call":
          indent[0] += 2
          print("-" * indent[0] + "> call function", frame.f_code.co_name)
      elif event == "return":
          print("<" + "-" * indent[0], "exit function", frame.f_code.co_name)
          indent[0] -= 2
      return tracefunc

device = 'cuda'

@torch.no_grad()
def main(name, model_name, batch_size, dir = None, use_gpu_sampling = False):
    BATCH_SIZE = batch_size
    enable_timers()
    clear_timers()
    infer_data = InferenceDataset(name, 0.1, force_reload=False, verbose=True)
    g = infer_data[0]
    logical_g = dgl.graph(g.edges())
    if use_gpu_sampling:
        logical_g = logical_g.to(device)
    else:
        g.pin_memory_()
        assert g.is_pinned()

    in_size = g.ndata["feat"].shape[1]
    out_size = infer_data.num_classes
    # Model goes on DEVICE
    model = load_model(model_name, in_size, out_size).to(device)
    model.eval()

    print(g)

    n = infer_data.trace_len // 2
    if infer_data._orig_name == 'reddit':
        n = infer_data.trace_len // 10

    for i in tqdm(range(0, n, BATCH_SIZE)):
        if i + BATCH_SIZE >= n:
            continue

        # TODO decide what to do if multiple infer requests for same node id
        # new_nid = infer_data.trace_nids[i:i+BATCH_SIZE]
        new_nid = []
        adj_nids = []
        sizes = []
        s = set()
        for idx in range(i, i + BATCH_SIZE):
            if infer_data.trace_nids[idx].item() not in s:
                adj_nids.append(infer_data.trace_edges[idx]["in"])
                sizes.append(infer_data.trace_edges[idx]["in"].shape[0])
                new_nid.append(infer_data.trace_nids[idx].item())
                s.add(infer_data.trace_nids[idx].item())
        
        new_nid = torch.tensor(new_nid)
        # if use_gpu_sampling:
        #     new_nid = new_nid.to(device)

        # assert(new_nid.shape == new_nid.unique().shape) 
        if BATCH_SIZE == 1:
            new_nid = new_nid.reshape(1)

        with Timer(name='total'):
            # TODO make MFG setup work with any batch size and number of layers
            # TODO see if this MFG setup can be done faster
            # TODO see GW FastToBlock https://github.com/gwsshs22/dgl/blob/infer-main/src/inference/graph_api.cc

            mfgs = []

            with Timer('sampling', track_cuda=use_gpu_sampling):
                # TODO test this batching very carefully
                # TODO reason to be suspicious: https://github.com/dmlc/dgl/issues/4512
                required_nodes = torch.cat(adj_nids)
                required_nodes_unique = required_nodes.unique()
                interleave_count = torch.tensor(sizes)
                # required_nodes = torch.cat([infer_data.trace_edges[idx]["in"] for idx in range(i, i+BATCH_SIZE)])
                # interleave_count = torch.tensor([infer_data.trace_edges[idx]["in"].shape[0] for idx in range(i, i+BATCH_SIZE)])

                # Create first layer message flow graph by looking at required neighbors
                all_seeds = torch.cat((required_nodes_unique, new_nid))

                if use_gpu_sampling:
                    # NOTE roughly 10x faster
                    frontier = dgl.sampling.sample_neighbors(logical_g, required_nodes_unique.to(device), -1)
                else:
                    frontier = dgl.sampling.sample_neighbors(logical_g, required_nodes_unique, -1)

                first_mfg = dgl.to_block(frontier, all_seeds) # Need to do cat here as should have target node

                # Create a message flow graph using the new edges
                mfg = dgl.graph((required_nodes, torch.repeat_interleave(new_nid, interleave_count)))
                last_mfg = dgl.to_block(mfg, new_nid)
            
                mfgs.append(first_mfg)
                mfgs.append(last_mfg)

            with Timer('dataloading', track_cuda=True):
                with Timer('feature gather'):
                    required_feats = first_mfg.ndata['_ID']['_N']
                    inputs = g.ndata['feat'][required_feats.cpu()]

                with Timer(name="CPU-GPU copy", track_cuda=True):
                    if device == 'cpu':
                        # TODO understand the overhead of this first access
                        inputs = mfgs[0].srcdata['feat']
                    else:
                        # NOTE When using GPU sampling the MFGs are already on GPU
                        # Graph.to(device) moves features as well
                        mfgs[0] = mfgs[0].to(device)
                        mfgs[1] = mfgs[1].to(device)

                        inputs = inputs.to(device)

            with Timer(name='model', track_cuda=True):
                x = model(mfgs, inputs)
                # Force sync
                x.cpu()

    print_timer_info()
    if dir != None:
        export_timer_info(f'{dir}/{model_name.upper()}', {'name': name, 'batch_size': batch_size})

if __name__ == '__main__':
    # main('ogbn-papers100M', 'gcn', 256)
    batch_sizes = [1, 64, 128, 256]

    use_gpu_sampling = True
    if use_gpu_sampling:
        path = 'benchmark/data/new_baseline_gpu'
        models = ['gcn']
        names = ['reddit', 'cora', 'ogbn-products']
    else:
        path = 'benchmark/data/new_baseline'
        models = ['gcn', 'sage', 'gat']
        names = ['reddit', 'cora', 'ogbn-products', 'ogbn-papers100M']

    for model in models:
        for name in names:
            for batch_size in batch_sizes:
                main(name=name, model_name=model, batch_size=batch_size, dir=path, use_gpu_sampling=use_gpu_sampling)
                gc.collect()
                gc.collect()
                gc.collect()
