from fast_inference.dataset import InferenceDataset
from fast_inference.models.gcn import GCN
from fast_inference.timer import enable_timers, Timer, print_timer_info
import dgl
import torch
from tqdm import tqdm

# TODO actually require feature movement, currently everything goes on one device
device = 'cpu'

@torch.no_grad()
def main():
    enable_timers()
    infer_data = InferenceDataset('cora', 0.1, force_reload=False, verbose=True)
    g = infer_data[0].to(device)
    
    in_size = g.ndata["feat"].shape[1]
    out_size = infer_data.num_classes
    # Model goes on DEVICE
    model = GCN(in_size, 16, out_size).to(device)
    model.eval()

    print(g)

    # Test static batch size of 1
    n = infer_data.trace_len
    for i in tqdm(range(n)):
        with Timer(name='total'):
            # TODO make MFG setup work with any batch size and number of layers
            # TODO see if this MFG setup can be done faster
            # TODO see GW FastToBlock https://github.com/gwsshs22/dgl/blob/infer-main/src/inference/graph_api.cc

            mfgs = []
            new_nid = infer_data.trace_nids[i].to(device)
            assert(g.in_edges(new_nid, form='eid').shape[0] == 0)
            assert(g.out_edges(new_nid, form='eid').shape[0] == 0)

            src_n = infer_data.trace_edges[i]["in"].to(device)
            
            # Create first layer message flow graph by looking at required neighbors
            frontier = dgl.sampling.sample_neighbors(g, src_n, -1)
            first_mfg = dgl.to_block(frontier, torch.cat((src_n, new_nid.reshape(1)))) # Need to do cat here as should have target node

            src_n = infer_data.trace_edges[i]["in"]
            dst_n = infer_data.trace_nids[i].expand(src_n.shape)

            # Create a message flow graph using the new edges
            mfg = dgl.graph((src_n, dst_n)).to(device)
            last_mfg = dgl.to_block(mfg, new_nid)
            
            # src_map = torch.arange(src_n.shape[0])
            # dst_map = torch.zeros(dst_n.shape)
            # print(src_map)
            # print(dst_map)
            # last_mfg = dgl.create_block((src_map, dst_map), num_src_nodes=src_n.shape[0], num_dst_nodes=dst_n.shape[0])

            # print(last_mfg)
            mfgs.append(first_mfg)
            mfgs.append(last_mfg)

            # Move features
            # mfgs[0].srcdata['feat'] = g.ndata['feat'][mfgs[0].srcdata[dgl.NID]]
            # mfgs[1].srcdata['feat'] = g.ndata['feat'][mfgs[1].srcdata[dgl.NID]]

            # print(mfgs[1].dstdata[dgl.NID])
            # print(g.ndata['feat'][mfgs[1].dstdata[dgl.NID]].shape)
            
            inputs = mfgs[0].srcdata['feat']
            mfgs[0].srcdata.pop('feat')
            mfgs[0].dstdata.pop('feat')
            with Timer(name='model'):
                model(mfgs, inputs)

    print_timer_info()

if __name__ == '__main__':
    main()