import torch
from pydgn.model.interface import ModelInterface
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, MessagePassing
from torch.nn import Linear, ReLU, Sequential
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


class GMDN(ModelInterface):
    '''
    Graph Mixture Density Network. The "predictor_class" implements is all our experts (with different parametrization)
    '''

    def _set_layer(self, emission, config):
        self.layer = GMDNEncoder(self.dim_node_features, self.dim_edge_features, self.no_experts, emission, config)

    def __init__(self, dim_node_features, dim_edge_features, dim_target, predictor_class, config):

        super().__init__()

        self.device = None
        self.dim_node_features = dim_node_features
        self.dim_edge_features = dim_edge_features
        self.dim_target = dim_target
        self.predictor_class = predictor_class
        self.layer_config = config

        self.no_experts = config['number_of_experts']

        self.max_epochs = config['epochs']

        self.no_layers = config['num_convolutional_layers']

        self.hidden_units = config['hidden_units']
        self.dirichlet_alpha = config['dirichlet_alpha']

        dim_experts_input = self.hidden_units * self.no_layers  # assume concatenation of all node embeddings
        emission = predictor_class(dim_experts_input, self.no_experts, dim_target, config)

        self._set_layer(emission, config)

    def to(self, device):
        self.device = device
        self.layer.to(device)
        return self

    def train(self, mode=True):
        super().train(mode)
        self.layer.train(mode)

    def eval(self):
        super().eval()
        self.layer.eval()

    def forward(self, data):
        return self.e_step(data)

    def e_step(self, data):

        # If training, E-step also accumulates statistics for the M-step!
        likely_labels, posterior_batch, objective, true_log_likelihood, prior_term, (params, p_Q_given_x) \
                                    = self.layer.e_step(data)

        # Detach as they are not used to back-propagate any gradient
        likely_labels, posterior_batch = likely_labels.detach(), posterior_batch.detach()

        prior_term = prior_term.detach()

        # Implements the interface required by TrainingEngine
        return likely_labels, posterior_batch, objective, true_log_likelihood, prior_term, (params, p_Q_given_x)

    def m_step(self):
        self.layer.m_step()

class GMDNTransition(torch.nn.Module):

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
        p_Q_given_x = torch.nn.functional.softmax(out_minus_max, dim=-1).clamp(1e-8, 1.)
        return p_Q_given_x, node_embeddings


class GMDNEncoder(torch.nn.Module):

    def __init__(self, dim_node_features, dim_edge_features, no_experts, emission, config):
        super().__init__()
        self.device = None

        self.dim_node_features = dim_node_features
        self.dim_edge_features = dim_edge_features
        self.no_experts = no_experts

        self.emission = emission

        self.hidden_units = config['hidden_units']
        self.dirichlet_alpha = config['dirichlet_alpha']
        self.dirichlet = torch.distributions.dirichlet.Dirichlet(torch.tensor([self.dirichlet_alpha] * self.no_experts,
                                                                              dtype=torch.float32))
        self.transition = GMDNTransition(dim_node_features, dim_edge_features, no_experts, config)  # neural convolution

    def to(self, device):
        if device is not None:
            self.device = device
            self.emission.to(device)
            self.transition.to(device)
            self.dirichlet.concentration = self.dirichlet.concentration.to(device)

    def forward(self, labels, graphs, edge_index, edge_attr, batch):
        return self.e_step(self, labels, graphs, edge_index, edge_attr, batch)

    def e_step(self, data):

        eps = 1e-8

        labels, graphs, edge_index, edge_attr, batch, = \
                                data.y, data.x, data.edge_index, data.edge_attr, data.batch

        # Perform the neural aggregation of neighbors to return the posterior P(Q=i | graph)
        # weight sharing: pass node embeddings produced by the gating network (which is a DGN) to each expert
        p_Q_given_x, node_embeddings = self.transition(graphs, edge_index, edge_attr, batch)

        distr_params = self.emission.get_distribution_parameters(node_embeddings, batch)
        likely_labels = self.emission.infer(p_Q_given_x, distr_params)  # inferring labels

        # E-step using true training labels
        emission_of_true_labels = self.emission.get_distribution_of_true_labels(labels, p_Q_given_x, distr_params)  # using true labels to compute the prob.
        unnorm_posterior_estimate = torch.mul(emission_of_true_labels, p_Q_given_x) + eps
        posterior_estimate = unnorm_posterior_estimate/unnorm_posterior_estimate.sum(dim=1, keepdim=True)

        true_log_likelihood = unnorm_posterior_estimate.sum(1).log().mean()
        complete_log_likelihood = torch.mean(torch.mul(posterior_estimate, unnorm_posterior_estimate.log()).sum(dim=1))

        prior_term = torch.mean(self.dirichlet.log_prob(p_Q_given_x), dim=0)

        objective = complete_log_likelihood + prior_term

        return likely_labels, p_Q_given_x, objective, true_log_likelihood, prior_term, (distr_params, p_Q_given_x)

    def m_step(self):
        pass  # all done by optimizer