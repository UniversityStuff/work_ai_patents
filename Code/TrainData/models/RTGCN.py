from torch_geometric.nn import GCNConv, HeteroConv, SAGEConv
from torch.nn import Linear
import torch.nn.functional as F
import torch



class RTGCN(torch.nn.Module):
    def __init__(self, in_channels, market, relation_path):
        super().__init__()

        # load graph
        self.graph = Graph(market=market, relation_path=relation_path)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        relation = torch.tensor(self.graph.relation, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        self.register_buffer('relation', relation)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = torch.nn.BatchNorm1d(in_channels * A.size(1))

        self.linear = torch.nn.Linear(self.relation.size(2),1)
        self.rt_gcn_networks = torch.nn.ModuleList((RTConv(in_channels, 8, kernel_size, 2),))

        # initialize parameters for edge importance weighting with "weighted" strategy
        self.edge_importance = torch.nn.ModuleList([torch.nn.Linear(self.relation.size(2),1) for i in self.rt_gcn_networks])

        # fcn for prediction
        self.fcn = torch.nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        # data normalization
        N, C, T, V = x.size()   # N: batch size, T: timestep, V: number of nodes, C: number of features
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(N, C, T, V)

        # relational-temporal graph convolution
        S, M, D = self.relation.size()
        for gcn, importance in zip(self.rt_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * (importance(self.relation.view(S*M, D)).view(S,M)+torch.eye(S).to(device)))

        # global pooling
        x = F.avg_pool2d(x,(x.size(2),1))
        x = x.view(N,-1,1,V)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), x.size(3))

        return x