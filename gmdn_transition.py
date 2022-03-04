import torch
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from gmdn_conv import GMDNConv


class GMDNTransition(torch.nn.Module):
    """
    Computes the vector of mixing weights
    """

    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__()

        self.dim_node_features = dim_node_features
        self.dim_edge_features = dim_edge_features
        self.dim_target = dim_target

        num_layers = config['num_convolutional_layers']
        hidden_units = config['hidden_units']

        if 'use_edge_attr' in config:
            self.use_edge_attr = config['use_edge_attr']
        else:
            self.use_edge_attr = False

        neighborhood_aggregation = config['neighborhood_aggregation']

        self.layers = torch.nn.ModuleList([])

        for i in range(num_layers):
            dim_input = dim_node_features if i == 0 else hidden_units
            if i == 0:
                conv = Sequential(Linear(dim_input, hidden_units, bias=False), ReLU(), Linear(hidden_units, hidden_units, bias=False), ReLU())  # fake conv
            else:
                conv = GMDNConv(
                    Sequential(Linear(dim_input, hidden_units, bias=False), ReLU(), Linear(hidden_units, hidden_units, bias=False), ReLU()),
                    aggr=neighborhood_aggregation, dim_edge_features=dim_edge_features if self.use_edge_attr else None)
            self.layers.append(conv)

        if config['aggregation'] == 'sum':
            self.aggregate = global_add_pool
        elif config['aggregation'] == 'mean':
            self.aggregate = global_mean_pool
        elif config['aggregation'] == 'max':
            self.aggregate = global_max_pool
        elif config['aggregation'] is None:  # for node classification
            self.aggregate = None

        self.fc_global = Linear(hidden_units * num_layers, hidden_units, bias=False)
        self.out = Linear(hidden_units, dim_target)

    def forward(self, x, edge_index, edge_attr, batch):
        x_all = []
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(x)
            else:
                x = layer(x, edge_index, edge_attr)
            x_all.append(x)
        node_embeddings = torch.cat(x_all, dim=1)

        if self.aggregate is not None:
            x = self.aggregate(x, batch)
        out = self.out(x)

        # Exp-normalize trick: subtract the maximum value
        max_vals, _ = torch.max(out, dim=-1, keepdim=True)
        out_minus_max = out - max_vals
        mixing_weights = torch.nn.functional.softmax(out_minus_max, dim=-1).clamp(1e-8, 1.)
        return mixing_weights, node_embeddings