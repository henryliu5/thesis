from ogb.nodeproppred import DglNodePropPredDataset
import dgl
import torch
import time
from torch.profiler import profile, ProfilerActivity, record_function

device = 'cuda'

if __name__ == '__main__':
    print(torch.cuda.memory_summary('cuda'))

    s = time.time()
    dataset = DglNodePropPredDataset('ogbn-papers100M')
    print('loading completed', time.time() - s)
    g, _ = dataset[0]
    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:

        logical_g = dgl.graph(g.edges())
        # logical_g = logical_g.to(device)
        logical_g.create_formats_()
        logical_g.pin_memory_()

        print('graph on device', logical_g)

        print(torch.cuda.memory_summary('cuda'))

        nids = torch.arange(300_000, device=device)
        with record_function('actual sampling'):
            # print(logical_g.out_degrees()[:10])
            print(logical_g.sample_neighbors(nids, -1))

    prof.export_chrome_trace("products_mem_trace.json")
    print(torch.cuda.memory_summary('cuda'))