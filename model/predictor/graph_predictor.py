import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from experiment.experiment import s2c


class GraphPredictor(torch.nn.Module):
    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__()
        self.dim_node_features = dim_node_features
        self.dim_edge_features = dim_edge_features
        self.dim_target = dim_target
        self.config = config

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        raise NotImplementedError('You need to implement this method!')


class IdentityGraphPredictor(GraphPredictor):

    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, config)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        return x, x


class LinearGraphPredictor(GraphPredictor):

    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, config)
        self.W = nn.Linear(dim_node_features, dim_target, bias=True)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = global_add_pool(x, batch)
        return self.W(x), x


class MLPGraphPredictor(nn.Module):

    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super(MLPGraphPredictor, self).__init__()

        hidden_units = config['hidden_units']

        self.fc_global = nn.Linear(dim_node_features, hidden_units)
        self.out = nn.Linear(hidden_units, dim_target)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = global_add_pool(x, batch)
        return self.out(F.relu(self.fc_global(x)))


class SimpleMLP(nn.Module):

    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__()

        hidden_units = config['hidden_units']
        self.fc_global = torch.nn.Linear(dim_node_features, hidden_units)
        self.out = torch.nn.Linear(hidden_units, dim_target)

    def forward(self, data):
        x = data.x.float()
        o = self.fc_global(x)
        out = self.out(F.relu(o))
        return out, x
