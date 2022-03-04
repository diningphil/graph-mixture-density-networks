import torch
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset


class GMDNConv(MessagePassing):
    """
    Inspired by the GINConv
    """
    def __init__(self, nn, eps=0, aggr='add', dim_edge_features=None, **kwargs):
        super(GMDNConv, self).__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.initial_eps = eps
        self.eps = torch.nn.Parameter(torch.Tensor([eps]))

        self.dim_edge_features = dim_edge_features
        if dim_edge_features is not None:
            self.edge_lin = Linear(dim_edge_features, 1)

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, edge_attr):
        out = self.nn((1. + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_attr))
        return out

    def message(self, x_j, edge_attr):
        if self.dim_edge_features is not None:
            return x_j if edge_attr is None else x_j*self.edge_lin(edge_attr)
        else:
            return x_j

