from typing import Tuple, Optional, List

import torch
from pydgn.model.interface import ReadoutInterface
from torch.distributions import Categorical, Independent, Binomial, Normal, MixtureSameFamily
from torch.nn import Identity, Linear, ReLU, Sequential
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


class GraphExpertEmission(ReadoutInterface):
    """
    STRUCTURE AGNOSTIC EMISSION, uses node embeddings. Implements all experts
    """

    def forward(self, node_embeddings: torch.tensor, batch: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        pass

    def __init__(self, dim_features, no_experts, dim_target, config):
        super().__init__(dim_features, 0, dim_target, config)

        self.no_experts = no_experts
        self.output_type = config['output_type']
        self.hidden_units = config['expert_hidden_units']
        self.dim_target = dim_target

        if config['aggregation'] == 'sum':
            self.aggregate = global_add_pool
        elif config['aggregation'] == 'mean':
            self.aggregate = global_mean_pool
        elif config['aggregation'] == 'max':
            self.aggregate = global_max_pool
        elif config['aggregation'] is None:  # for node classification
            self.aggregate = None

        if 'binomial' in self.output_type:
            assert dim_target == 1, "Implementation works with a single dim regression problem for now"
            self.output_activation = Identity()  # emulate bernoulli
            self.node_transform = Identity()
            self.final_transform = Linear(dim_features, self.no_experts * 2, bias=False)
        elif 'gaussian' in self.output_type:
            self.output_activation = Identity()  # emulate gaussian (needs variance as well
            # Need independent parameters for the variance
            if self.hidden_units > 0:
                self.node_transform = Sequential(Linear(dim_features, self.hidden_units * self.no_experts * 2), ReLU())
                self.final_transform = Linear(self.hidden_units * self.no_experts * 2, self.no_experts * 2 * dim_target)
            else:
                self.node_transform = Identity()
                self.final_transform = Linear(dim_features, self.no_experts * 2 * dim_target)
        else:
            raise NotImplementedError(f'Activation {self.output_type} unrecognized, use binomal, gaussian.')

    def get_distribution_parameters(self, node_embeddings, batch):

        if self.aggregate is not None:
            graph_embeddings = self.aggregate(self.node_transform(node_embeddings), batch)
            out = self.output_activation(self.final_transform(graph_embeddings))
        else:
            out = self.output_activation(self.final_transform(self.node_transform(node_embeddings)))



        if 'binomial' in self.output_type:
            params = torch.reshape(out, [-1, self.no_experts, 2])  # ? x no_experts x K
            # first parameter not used here
            _, p = torch.round(torch.relu(params[:, :, 0])) + 1, torch.sigmoid(params[:, :, 1])
            n = global_add_pool(torch.ones(node_embeddings.shape[0], self.no_experts).to(node_embeddings.device), batch)

            distr_params = (n, p)
        elif 'gaussian' in self.output_type:
            # Assume isotropic gaussians
            params = torch.reshape(out, [-1, self.no_experts, 2, self.dim_target])  # ? x no_experts x 2 x F
            mu, var = params[:, :, 0, :], params[:, :, 1, :]

            var = torch.nn.functional.softplus(var) + 1e-8
            # F is assumed to be 1 for now, add dimension to F

            distr_params = (mu, var)   # each has shape ? x no_experts X F

        return distr_params

    def get_distribution_of_true_labels(self, labels, p_Q_given_x, distr_params):
        """
        For each expert i, returns the probability associated to a specific label.
        """
        eps = 1e-12

        if 'binomial' in self.output_type:
            n, p = distr_params

            # Assume 1 for now
            if len(n.shape) == 2:
                n = n.unsqueeze(2)  # add output feature dim
                # distr_params is now [samples, no_experts, 1=no_features]
            if len(p.shape) == 2:
                p = p.unsqueeze(2)  # add output feature dim

            mix = Categorical(p_Q_given_x)
            comp = Independent(Binomial(n, p), 1)
            pmm = MixtureSameFamily(mix, comp)

            if len(labels.shape) == 1:
                labels = labels.unsqueeze(1)  # add output feature dim

            x = pmm._pad(labels)
            emission_of_true_labels = pmm.component_distribution.log_prob(x.float()).exp() + eps # [samples, experts]

        elif 'gaussian' in self.output_type:
            mu, var = distr_params

            mix = Categorical(p_Q_given_x)
            comp = Independent(Normal(loc=mu, scale=var), 1)  # mu/var have shape [samples, experts, features]
            gmm = MixtureSameFamily(mix, comp)

            # labels has shape [samples, features]
            x = gmm._pad(labels)
            emission_of_true_labels = gmm.component_distribution.log_prob(x).exp() # [samples, experts], one prob for each expert

        return emission_of_true_labels

    def infer(self, p_Q, distr_params):
        """
        Get the value associated with the most confident predictor_class
        :param p_Q: tensor of size ?xC
        :return:
        """
        # We simply compute the conditional mean P(y|x) = \sum_i P(y|Q=i)P(Q=i|x) for each node
        if 'binomial' in self.output_type:
            n, p = distr_params
            mean = n*p
            tmp = torch.mul(p_Q, mean)  # (? x no_experts)
            inferred_y = tmp.sum(1, keepdim=True)  # ? x 1
        elif 'gaussian' in self.output_type:
            mu, var = distr_params
            tmp = torch.mul(p_Q.unsqueeze(2), mu)  # (? x no_experts, F)
            inferred_y = torch.sum(tmp, dim=1)  # ? x F
        return inferred_y  # avoid recomputing params
