from fast_inference.dataset import InferenceDataset
from fast_inference.models.gcn import GCN
from fast_inference.timer import enable_timers, Timer, print_timer_info
import dgl
import torch
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
from dgl.dataloading import MultiLayerFullNeighborSampler
import time

device = 'cuda'

if __name__ == '__main__':
    print(torch.cuda.memory_summary('cuda'))
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    device='cuda'
    s = time.time()
    infer_data = InferenceDataset('ogbn-papers100M', 0.01, force_reload=False, verbose=True)
    print('loading completed', time.time() - s)
    infer_data.create_inference_trace(subgraph_bias=None)

    g = infer_data[0]
    print('graph not on cuda')
    logical_g = dgl.graph(g.edges())
    logical_g.create_formats_()
    logical_g = logical_g.to(device)
    print('graph on cuda')

    print(torch.cuda.memory_summary('cuda'))
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
    
    print(logical_g.out_degrees()[:10])
    nids = torch.arange(300_000, device=device)
    print(logical_g.sample_neighbors(nids, -1))