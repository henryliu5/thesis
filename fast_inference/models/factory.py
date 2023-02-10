from .gat import GAT
from .gcn import GCN
from .sage import SAGE

# TODO make this more customizable - probably want to add tests
def load_model(model_type, num_inputs, num_outputs):
    if model_type == 'gcn':
        model = GCN(num_inputs, num_outputs)
    elif model_type == 'sage':
        model = SAGE(num_inputs, num_outputs)
    elif model_type == 'gat':
        model = GAT(num_inputs, num_outputs)
    else:
        print(f"Unknown model_type: {model_type}")
        exit(-1)
    return model