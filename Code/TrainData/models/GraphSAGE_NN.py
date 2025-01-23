from torch_geometric.nn import SAGEConv
from torch.nn import Linear
import torch.nn.functional as F
import torch

class GraphSAGE_NN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, type_pooling, num_nodes):
        super(GraphSAGE_NN, self).__init__()

        # First layer of GraphSage
        lst_GraphSAGE = [SAGEConv(input_dim, hidden_dim, aggr = type_pooling)]

        # Hidden layers of GraphSage
        for i in range(num_layers - 2):
          lst_GraphSAGE.append(SAGEConv(hidden_dim, hidden_dim, aggr = type_pooling))
        lst_GraphSAGE.append(SAGEConv(hidden_dim, output_dim, aggr = type_pooling))

        # Final layer of GraphSage
        self.convs = torch.nn.ModuleList(lst_GraphSAGE)
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(num_features = hidden_dim * num_nodes, affine = False) for i in range(num_layers - 1)])

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i in range(len(self.convs) - 1):
          x = self.convs[i].forward(x, adj_t)   # graph convolution
          N,T,V,C =  x.size()   # N: batch size, T: timestep, V: number of nodes, C: number of features
          x = torch.permute(x, (0,2,3,1))
          x = x.view(N, V * C, T)
          x = self.bns[i](x)   # normalization
          x = torch.nn.functional.relu(x)   # activation
          x = torch.nn.functional.dropout(x,self.dropout)   # dropout
          x = x.view(N, V, C, T)
          x = x.permute(0, 3, 1, 2).contiguous()

        out = self.convs[-1].forward(x, adj_t)
        return out