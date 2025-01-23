from torch_geometric.nn import GCNConv, HeteroConv, SAGEConv
from HeteroGCN import HeteroGCN
from torch.nn import Linear
import torch.nn.functional as F
import torch


class RTConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, num_nodes=854):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        #######################################################
        # Change different GNN layers here:

        # Single layer GCN
        #self.gcn = GCNConv(-1, out_channels)

        # Multi layers GCN (3 layers)
        self.gcn = HeteroGCN(-1, int(out_channels*1.5), out_channels, 3, 0.5, num_nodes)

        # Single layer GraphSage
        #self.gcn = SAGEConv(-1, out_channels)

        # Multi layers GraphSage (3 layers)
        #self.gcn = GraphSAGE_NN(-1, int(out_channels*1.5), out_channels, 3, 0.5, "mean", num_nodes)
        #######################################################

        # Temporal convolution
        self.tcn = torch.nn.Sequential(
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Dropout(dropout, inplace=True),
        )

        self.residual = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels,kernel_size=1, stride=(stride, 1)),
            torch.nn.BatchNorm2d(out_channels),
        )

        self.relu = torch.nn.LeakyReLU(inplace=True,negative_slope=0.2)

    def forward(self, x, A):
        res = self.residual(x) # for processing the residual component
        x = x.permute(0,2,3,1)
        # relational graph convolution
        x = self.gcn(x, A[0].to_sparse().indices())
        x = x.permute(0,3,1,2)
        # temporal convolution and residual connection
        x = self.tcn(x) + res

        return self.relu(x), A