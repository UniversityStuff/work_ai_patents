from torch_geometric.nn import GCNConv, HeteroConv, SAGEConv
from torch.nn import Linear
import torch.nn.functional as F
import torch

# Code from https://medium.com/stanford-cs224w/temporal-graph-learning-for-stock-prediction-58429696f482

class HeteroGCN(torch.nn.Module):
    def __init__(self, hidden_channels, data):
        super(HeteroGCN, self).__init__()
        torch.manual_seed(42)

        # Initialize the layers
        self.convs = HeteroConv({
            ('patents', 'same_classification', 'patents'): GCNConv(data['patents'].x.size(1), hidden_channels)
        }, aggr='sum')
        self.lin = Linear(hidden_channels, data['patents'].x.size(1))

    def forward(self, x_dict, edge_index_dict):
        # First Message Passing Layer (Transformation)
        x_dict = self.convs(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=0.5, training=self.training) for key, x in x_dict.items()}

        # Output layer
        x_dict['patents'] = self.lin(x_dict['patents'])
        return x_dict
    
