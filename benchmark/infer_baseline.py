from fast_inference.dataset import InferenceDataset
from fast_inference.models.gcn import GCN
from fast_inference.timer import enable_timers, Timer, print_timer_info, export_timer_info, clear_timers
import dgl
import torch
from tqdm import tqdm

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
def main(name, batch_size):
    BATCH_SIZE = batch_size
    enable_timers()
    clear_timers()
    infer_data = InferenceDataset(name, 0.1, force_reload=False, verbose=True)
    g = infer_data[0]
    
    in_size = g.ndata["feat"].shape[1]
    out_size = infer_data.num_classes
    # Model goes on DEVICE
    model = GCN(in_size, 16, out_size).to(device)
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
        # assert(new_nid.shape == new_nid.unique().shape) 
        if BATCH_SIZE == 1:
            new_nid = new_nid.reshape(1)

        with Timer(name='total'):
            # TODO make MFG setup work with any batch size and number of layers
            # TODO see if this MFG setup can be done faster
            # TODO see GW FastToBlock https://github.com/gwsshs22/dgl/blob/infer-main/src/inference/graph_api.cc

            mfgs = []

            with Timer('(cpu) sampling'):
                # TODO test this batching very carefully
                # TODO reason to be suspicious: https://github.com/dmlc/dgl/issues/4512
                required_nodes = torch.cat(adj_nids)
                interleave_count = torch.tensor(sizes)
                # required_nodes = torch.cat([infer_data.trace_edges[idx]["in"] for idx in range(i, i+BATCH_SIZE)])
                # interleave_count = torch.tensor([infer_data.trace_edges[idx]["in"].shape[0] for idx in range(i, i+BATCH_SIZE)])

                # Create first layer message flow graph by looking at required neighbors
                frontier = dgl.sampling.sample_neighbors(g, required_nodes.unique(), -1)
                first_mfg = dgl.to_block(frontier, torch.cat((required_nodes.unique(), new_nid))) # Need to do cat here as should have target node

                # Create a message flow graph using the new edges
                mfg = dgl.graph((required_nodes, torch.repeat_interleave(new_nid, interleave_count)))
                last_mfg = dgl.to_block(mfg, new_nid)
            
            mfgs.append(first_mfg)
            mfgs.append(last_mfg)

            with Timer(name="CPU-GPU copy", track_cuda=True):
                if device == 'cpu':
                    # TODO understand the overhead of this first access
                    inputs = mfgs[0].srcdata['feat']
                else:
                    # Graph.to(device) moves features as well
                    mfgs[0] = mfgs[0].to(device)
                    mfgs[1] = mfgs[1].to(device)
                    inputs = mfgs[0].srcdata['feat']

            mfgs[0].srcdata.pop('feat')
            mfgs[0].dstdata.pop('feat')

            with Timer(name='model', track_cuda=True):
                model(mfgs, inputs)

    print_timer_info()
    export_timer_info(f'benchmark/data/timing_breakdown', {'name': name, 'batch_size': batch_size})

if __name__ == '__main__':
    names = ['reddit', 'cora', 'ogbn-products', 'ogbn-papers100M']
    batch_sizes = [128]
    for name in names:
        for batch_size in batch_sizes:
            main(name=name, batch_size=batch_size)