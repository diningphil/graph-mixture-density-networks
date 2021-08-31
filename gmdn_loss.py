import torch
import torch.nn.functional as F
from pydgn.training.callback.loss import Loss
from torch.distributions.normal import Normal


class GMDNLoss(Loss):

    # Simply ignore targets
    def forward(self, targets, *outputs):
        likelihood = outputs[2]
        return likelihood

    def on_backward(self, state):
        loss = -state.batch_loss[self.__name__]
        loss.backward()


class LinkPredictionLoss(Loss):
    __name__ = 'Link Prediction Loss'

    def __init__(self):
        super().__init__()

    def forward(self, targets, *outputs):
        node_embs = outputs[1]
        _, pos_edges, neg_edges = targets[0]

        loss_edge_index = torch.cat((pos_edges, neg_edges), dim=1)
        loss_target = torch.cat((torch.ones(pos_edges.shape[1]),
                                  torch.zeros(neg_edges.shape[1])))

        # Taken from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/link_pred.py
        x_j = torch.index_select(node_embs, 0, loss_edge_index[0])
        x_i = torch.index_select(node_embs, 0, loss_edge_index[1])
        link_logits = torch.einsum("ef,ef->e", x_i, x_j)
        loss =  torch.nn.functional.binary_cross_entropy_with_logits(link_logits, loss_target)
        return loss


class L2GMMLinkPredictionLoss(Loss):

    def __init__(self):
        super().__init__()

    # This is two orders of magnitude faster than the naive version (see Github commits)
    # Taken from http://kyoustat.com/pdf/note004gmml2.pdf and implemented in matrix form
    @staticmethod
    def compute_loss(mu, var, weights, loss_edge_index):

        # Batched outer product for 1-D gaussians
        # A and B coefficients
        # you index alphas_betas with a node and you get the outer product for that node
        alphas_betas = torch.bmm(weights.unsqueeze(2), weights.unsqueeze(1))
        # equivalent but slower
        #weights_einsum = torch.einsum('bi,bj->bij', (weights, weights))

        # C coefficients (necessarily dependent on edges)
        gammas = torch.bmm(weights[loss_edge_index[0]].unsqueeze(2), weights[loss_edge_index[1]].unsqueeze(1))

        # expand does not copy data
        MU = mu.unsqueeze(2).expand(mu.shape[0], mu.shape[1], mu.shape[1])  # (nodes, experts, experts)
        MU_T = MU.transpose(2, 1)

        # This is the batched pairwise sum of elements (works with 1-D)
        VAR = var.unsqueeze(2) + var.unsqueeze(1)  # (nodes, experts, experts)

        # :P
        A_normal = Normal(MU.reshape(-1), VAR.reshape(-1))
        # We istantiate A and B from A_B
        A_B = torch.mul(alphas_betas, A_normal.log_prob(MU_T.reshape(-1)).exp().reshape(MU.shape)).sum((1,2))

        A, B = A_B[loss_edge_index[0]], A_B[loss_edge_index[1]]

        # We do the same as before, but considering edges and changing MU and MU_T and VAR with those referring to two distinct distributions
        MU_1, MU_2 = MU[loss_edge_index[0]], MU_T[loss_edge_index[1]]  # (edges, experts, experts)

        VAR = var[loss_edge_index[0]].unsqueeze(2) + var[loss_edge_index[1]].unsqueeze(1)  # (edges, experts, experts)
        C_normal = Normal(MU_1.reshape(-1), VAR.reshape(-1))
        C = torch.mul(gammas, C_normal.log_prob(MU_2.reshape(-1)).exp().reshape(MU_2.shape)).sum((1,2))

        #print(mu, var, weights)
        #print(A + B - 2*C)
        #assert False

        squared_distance_clean = A + B - 2.*C
        # Add a relu to compensate for small numerical errors
        squared_distance = torch.relu(squared_distance_clean) + 1e-8  # this constant accelerates initial convergence!

        return squared_distance

    def forward(self, targets, *outputs):
        """
        This works for mixtures of 1-D Gaussians
        """
        weights, mu, var = outputs[1][0], outputs[1][1], outputs[1][2]
        node_embs = outputs[1][3]
        emission_weight = outputs[1][4]
        prior_term = outputs[1][5]

        _, pos_edges, neg_edges = targets[0]

        loss_edge_index = torch.cat((pos_edges, neg_edges), dim=1)

        squared_distance = L2GMMLinkPredictionLoss.compute_loss(mu, var, weights, loss_edge_index)
        distance = torch.sqrt(squared_distance)

        #pos_distance = distance[:pos_edges.shape[1]]
        #neg_distance = distance[pos_edges.shape[1]:]
        #return pos_distance.mean() - torch.log(1. + neg_distance).mean()
        loss_target = torch.cat((torch.zeros(pos_edges.shape[1]),
                                 torch.ones(neg_edges.shape[1])+1.))
        return torch.nn.functional.l1_loss(distance, loss_target)


class JeffreyGMMLinkPredictionLoss(Loss):

    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_loss(mu, var, weights, loss_edge_index):

        mu_src, mu_dst = mu[loss_edge_index[0]], mu[loss_edge_index[1]]
        var_src, var_dst = var[loss_edge_index[0]], var[loss_edge_index[1]]
        weights_src, weights_dst = weights[loss_edge_index[0]], weights[loss_edge_index[1]]

        diff_mu = mu_src - mu_dst
        kl_src_dst = torch.log(torch.sqrt(var_dst)/torch.sqrt(var_src)) + (var_src +  torch.mul(diff_mu, diff_mu))/(2.*var_dst) - 0.5
        kl1 = torch.mul(weights_src, kl_src_dst)

        diff_mu = mu_dst - mu_src
        kl_dst_src = torch.log(torch.sqrt(var_src)/torch.sqrt(var_dst)) + (var_dst +  torch.mul(diff_mu, diff_mu))/(2.*var_src) - 0.5
        kl2 = torch.mul(weights_dst, kl_dst_src)

        # Jeffrey's divergence: https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
        # With alpha-JS divergence with alpha = 1  (technically it's half of the Jeffrey's Divergence, but it does not change the optimization problem)
        return ((kl1 + kl2)/2).sum(dim=1)
        #return kl1.sum(dim=1)

    def forward(self, targets, *outputs):
        """
        This works for mixtures of 1-D Gaussians
        """
        weights, mu, var = outputs[1][0], outputs[1][1], outputs[1][2]
        node_embs = outputs[1][3]
        emission_weight = outputs[1][4]
        prior_term = outputs[1][5]

        _, pos_edges, neg_edges = targets[0]

        loss_edge_index = torch.cat((pos_edges, neg_edges), dim=1)
        loss_target = torch.cat((torch.zeros(pos_edges.shape[1]),
                                 torch.ones(neg_edges.shape[1])+1.))

        distance = JeffreyGMMLinkPredictionLoss.compute_loss(mu, var, weights, loss_edge_index)

        return torch.nn.functional.l1_loss(distance, loss_target)
        #pos_distance = distance[:pos_edges.shape[1]]
        #neg_distance = distance[pos_edges.shape[1]:]


class BhattacharyyaGMMLinkPredictionLoss(Loss):

    def __init__(self):
        super().__init__()

    # https://en.wikipedia.org/wiki/Bhattacharyya_distance
    @staticmethod
    def compute_loss(mu, var, weights, loss_edge_index):

        mu_src, mu_dst = mu[loss_edge_index[0]], mu[loss_edge_index[1]]
        var_src, var_dst = var[loss_edge_index[0]], var[loss_edge_index[1]]
        weights_src, weights_dst = weights[loss_edge_index[0]], weights[loss_edge_index[1]]

        diff_mu = mu_src - mu_dst
        sigma = 0.5*(var_src + var_dst)
        b_src_dst = (1./8.)*torch.mul(diff_mu, diff_mu)/(sigma) + 0.5*torch.log(sigma/torch.sqrt(torch.mul(var_src, var_dst)))
        b = torch.mul(torch.sqrt(torch.mul(weights_src, weights_dst)), b_src_dst)

        return b.sum(dim=1)

    def forward(self, targets, *outputs):
        """
        This works for mixtures of 1-D Gaussians
        """
        weights, mu, var = outputs[1][0], outputs[1][1], outputs[1][2]
        node_embs = outputs[1][3]
        emission_weight = outputs[1][4]
        prior_term = outputs[1][5]

        _, pos_edges, neg_edges = targets[0]

        loss_edge_index = torch.cat((pos_edges, neg_edges), dim=1)
        loss_target = torch.cat((torch.zeros(pos_edges.shape[1]),
                                 torch.ones(neg_edges.shape[1])+1.))

        distance = BhattacharyyaGMMLinkPredictionLoss.compute_loss(mu, var, weights, loss_edge_index)

        pos_distance = distance[:pos_edges.shape[1]]
        neg_distance = distance[pos_edges.shape[1]:]
        return torch.nn.functional.l1_loss(distance, loss_target)

        #loss_target = torch.cat((torch.ones(pos_edges.shape[1]),
        #                         torch.zeros(neg_edges.shape[1])-1.))
        #return torch.nn.functional.hinge_embedding_loss(distance, loss_target, margin=2.0)


class DotProductGMMLinkPredictionLoss(Loss):

    def __init__(self):
        super().__init__()

    def forward(self, targets, *outputs):
        """
        This works for mixtures of 1-D Gaussians
        """
        weights, mu = outputs[1][0], outputs[1][1]
        _, pos_edges, neg_edges = targets[0]

        # Do not use the weights (just transform node embeddings and then apply dot product)
        node_embs = mu
        loss_edge_index = torch.cat((pos_edges, neg_edges), dim=1)
        loss_target = torch.cat((torch.ones(pos_edges.shape[1]),
                                  torch.zeros(neg_edges.shape[1])))
        # Taken from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/link_pred.py
        x_j = torch.index_select(node_embs, 0, loss_edge_index[0])
        x_i = torch.index_select(node_embs, 0, loss_edge_index[1])
        link_logits = torch.einsum("ef,ef->e", x_i, x_j)
        loss =  torch.nn.functional.binary_cross_entropy_with_logits(link_logits, loss_target)
        return loss
