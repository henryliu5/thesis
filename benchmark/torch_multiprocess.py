import torch
from torch.profiler import profile, record_function, ProfilerActivity
# # doesn't work with either sharing strategy (alternative is 'file_system')
# torch.multiprocessing.set_sharing_strategy('file_descriptor') 
# a = []
# for i in range(100_000):
#     x = torch.ones(1)
#     x.share_memory_()
#     a.append(x)

if __name__ == '__main__':
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        s0 = torch.cuda.Stream(device=torch.device('cuda', 0))
        s0_alt = torch.cuda.Stream(device=torch.device('cuda', 0))
        s0_alt_cpu = torch.cuda.Stream(device=torch.device('cuda', 0))

        s1 = torch.cuda.Stream(device=torch.device('cuda', 1))
        s1_alt = torch.cuda.Stream(device=torch.device('cuda', 1))
        s1_alt_cpu = torch.cuda.Stream(device=torch.device('cuda', 1))

        cpu = torch.rand((100_000, 128), pin_memory=True)
        for i in range(100):
            with torch.cuda.stream(s0):
                x = torch.rand((9000, 9000), device=torch.device('cuda', 0))
                x1 = x.to(torch.device('cuda', 1))

            with torch.cuda.stream(s0_alt):
                # Work on GPU can happen during P2P transfer
                x[:1000,:1000].matmul(x[:1000,:1000])

            with torch.cuda.stream(s0_alt_cpu):
                # including CPU-GPU transfer
                cpu.to(torch.device('cuda', 0), non_blocking=True)

            with torch.cuda.stream(s1):
                # Peer to peer transfers can happen in parallel if different directions on NVLink
                x2 = x1.to(torch.device('cuda', 0))

            with torch.cuda.stream(s1_alt):
                # Peer to peer transfers can happen in parallel, but in the same direction they will be serialized
                x2 = x1.to(torch.device('cuda', 0))

            with torch.cuda.stream(s1_alt_cpu):
                # including CPU-GPU transfer
                cpu.to(torch.device('cuda', 1), non_blocking=True)
            
    
    prof.export_chrome_trace('p2p_bandwidth_trace.json')
