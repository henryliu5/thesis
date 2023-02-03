import torch

class FeatureServer:
    def __init__(self):
        pass
    
    def get_features(self, node_ids: torch.Tensor):
        """Get features for a list of nodes.

        Features are fetched from GPU memory if cached, otherwise from CPU memory.

        Args:
            node_ids (torch.Tensor): A 1-D tensor of node IDs.
        """
        pass