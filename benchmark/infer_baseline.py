from fast_inference.dataset import InferenceDataset
from fast_inference.models.gcn import GCN
from fast_inference.timer import enable_timers, Timer, print_timer_info
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
def main():
    enable_timers()
    infer_data = InferenceDataset('ogbn-products', 0.1, force_reload=False, verbose=True)
    g = infer_data[0]
    
    in_size = g.ndata["feat"].shape[1]
    out_size = infer_data.num_classes
    # Model goes on DEVICE
    model = GCN(in_size, 16, out_size).to(device)
    model.eval()

    print(g)

    # Test static batch size of 1
    n = infer_data.trace_len // 2
    if infer_data._orig_name == 'reddit':
        n = 200
    for i in tqdm(range(n)):
        with Timer(name='total'):
            # TODO make MFG setup work with any batch size and number of layers
            # TODO see if this MFG setup can be done faster
            # TODO see GW FastToBlock https://github.com/gwsshs22/dgl/blob/infer-main/src/inference/graph_api.cc

            mfgs = []
            new_nid = infer_data.trace_nids[i]
            src_n = infer_data.trace_edges[i]["in"]
            assert(g.in_edges(new_nid, form='eid').shape[0] == 0)
            assert(g.out_edges(new_nid, form='eid').shape[0] == 0)

            with Timer('(cpu) sampling'):
                # Create first layer message flow graph by looking at required neighbors
                frontier = dgl.sampling.sample_neighbors(g, infer_data.trace_edges[i]["in"], -1)
                first_mfg = dgl.to_block(frontier, torch.cat((src_n, new_nid.reshape(1)))) # Need to do cat here as should have target node

                # Create a message flow graph using the new edges
                mfg = dgl.graph((src_n, new_nid.expand(src_n.shape)))
                last_mfg = dgl.to_block(mfg, new_nid)
            
            # src_map = torch.arange(src_n.shape[0])
            # dst_map = torch.zeros(dst_n.shape)
            # last_mfg = dgl.create_block((src_map, dst_map), num_src_nodes=src_n.shape[0], num_dst_nodes=dst_n.shape[0])

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

if __name__ == '__main__':
    main()