from fast_inference.timer import Timer
import torch
import dgl

class InferenceSampler:
    ''' Generates MFGs for new inference requests
    '''
    def __init__(self, g):
        self.g = g

    def sample(self, nids, edges, *, use_gpu_sampling, fix_unique=False):
        # TODO make MFG setup work with any batch size and number of layers
        with Timer('sampling', track_cuda=use_gpu_sampling):
            if use_gpu_sampling:
                device = 'cuda'
            else:
                device = 'cpu'
            mfgs = []
            batch_size = nids.shape[0]

            if fix_unique:
                new_nid = []
                adj_nids = []
                sizes = []
                s = set()
                for idx in range(batch_size):
                    if nids[idx].item() not in s:
                        adj_nids.append(edges[idx]["in"])
                        sizes.append(edges[idx]["in"].shape[0])
                        new_nid.append(nids[idx].item())
                        s.add(nids[idx].item())
                new_nid = torch.tensor(new_nid)

                required_nodes = torch.cat(adj_nids)
                interleave_count = torch.tensor(sizes)
            else:
                # TODO test this batching very carefully
                # TODO reason to be suspicious: https://github.com/dmlc/dgl/issues/4512
                interleave_count = torch.tensor([edges[idx]["in"].shape[0] for idx in range(batch_size)])
                required_nodes = torch.empty(interleave_count.sum(), dtype=torch.int64, pin_memory=True)
                torch.cat([edges[idx]["in"] for idx in range(batch_size)], out=required_nodes)
                interleave_count = interleave_count.to(device)
                required_nodes = required_nodes.to(device)
                new_nid = nids.to(device)

                # required_nodes = torch.cat([edges[idx]["in"] for idx in range(batch_size)])
                # required_nodes = required_nodes.to(device)
                # interleave_count = torch.tensor([edges[idx]["in"].shape[0] for idx in range(batch_size)], device=device)
                # new_nid = nids.to(device)

            required_nodes_unique = required_nodes.unique()
            # assert(new_nid.shape == orig_new_nid.shape)
            # assert(new_nid.shape == new_nid.unique().shape) 
            if batch_size == 1:
                new_nid = new_nid.reshape(1)

            # TODO see if this MFG setup can be done faster
            # TODO see GW FastToBlock https://github.com/gwsshs22/dgl/blob/infer-main/src/inference/graph_api.cc
            
            # Create first layer message flow graph by looking at required neighbors
            all_seeds = torch.cat((required_nodes_unique, new_nid))

            with Timer('dgl sample neighbors'):
                if use_gpu_sampling:
                    # NOTE roughly 10x faster
                    assert (self.g.device != torch.device('cpu'))
                    frontier = dgl.sampling.sample_neighbors(self.g, required_nodes_unique.to('cuda'), -1)
                else:
                    frontier = dgl.sampling.sample_neighbors(self.g, required_nodes_unique, -1)

            with Timer('dgl first to block'):
                first_mfg = dgl.to_block(frontier, all_seeds) # Need to do cat here as should have target node

            with Timer('dgl create mfg graph'):
                # Create a message flow graph using the new edges
                mfg = dgl.graph((required_nodes, torch.repeat_interleave(new_nid, interleave_count)))

            with Timer('dgl last mfg to block'):
                last_mfg = dgl.to_block(mfg, new_nid)
        
            mfgs.append(first_mfg)
            mfgs.append(last_mfg)

            return mfgs