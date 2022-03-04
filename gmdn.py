from typing import Tuple, Optional, List

import torch
from pydgn.model.interface import ModelInterface
from torch_geometric.data import Batch

from gmdn_transition import GMDNTransition


class GMDN(ModelInterface):
    """
    Graph Mixture Density Network. The "readout_class" implements is all our experts (with different parametrization)
    """

    def __init__(self, dim_node_features, dim_edge_features, dim_target, readout_class, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, readout_class, config)

        self.device = None
        self.readout_class = readout_class

        self.no_experts = config['number_of_experts']
        self.max_epochs = config['epochs']
        self.no_layers = config['num_convolutional_layers']
        self.hidden_units = config['hidden_units']
        self.dirichlet_alpha = config['dirichlet_alpha']

        dim_experts_input = self.hidden_units * self.no_layers  # assume concatenation of all node embeddings
        emission = readout_class(dim_experts_input, self.no_experts, dim_target, config)

        self.encoder = GMDNEncoder(self.dim_node_features, self.dim_edge_features, self.no_experts, emission, config)

    def to(self, device):
        if device is not None:
            self.encoder.to(device)

    def forward(self, data: Batch) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        likely_labels, posterior_batch, objective, true_log_likelihood, prior_term = self.encoder.forward(data)

        # Detach as they are not used to back-propagate any gradient
        likely_labels, posterior_batch = likely_labels.detach(), posterior_batch.detach()
        prior_term = prior_term.detach()

        # Implements the interface required by TrainingEngine
        return likely_labels, posterior_batch, [objective, true_log_likelihood, prior_term]


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

    def forward(self, data):
        eps = 1e-8
        labels, graphs, edge_index, edge_attr, batch, = \
            data.y, data.x, data.edge_index, data.edge_attr, data.batch

        # Perform the neural aggregation of neighbors to return the posterior P(Q=i | graph)
        # weight sharing: pass node embeddings produced by the gating network (which is a DGN) to each expert
        mixing_weights, node_embeddings = self.transition(graphs, edge_index, edge_attr, batch)

        distr_params = self.emission.get_distribution_parameters(node_embeddings, batch)

        likely_labels = self.emission.infer(mixing_weights, distr_params)  # inferring labels

        # E-step using true training labels
        p_X_given_Q = self.emission.get_distribution_of_true_labels(labels, mixing_weights, distr_params)

        # Bayes theorem
        unnorm_posterior_estimate = torch.mul(p_X_given_Q, mixing_weights) + eps
        posterior_estimate = unnorm_posterior_estimate / unnorm_posterior_estimate.sum(dim=1, keepdim=True)

        # weighted sum across all components (computing the marginal P(X) basically)
        log_likelihood = unnorm_posterior_estimate.sum(1).log().mean()
        complete_log_likelihood = torch.mean(torch.mul(posterior_estimate, unnorm_posterior_estimate.log()).sum(dim=1))

        prior_term = torch.mean(self.dirichlet.log_prob(mixing_weights), dim=0)
        objective = complete_log_likelihood + prior_term

        return likely_labels, mixing_weights, objective, log_likelihood, prior_term
