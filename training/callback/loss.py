import time
import operator
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.nn.modules.loss import MSELoss, L1Loss, BCELoss
from training.event.handler import EventHandler


class Loss(nn.Module, EventHandler):
    """
    Loss is the main event handler for loss metrics. Other losses can easily subclass by implementing the forward
    method, though sometimes more complex implementations are required.
    """
    __name__ = "loss"
    op = operator.lt  # less than to determine improvement

    def __init__(self):
        super().__init__()
        self.batch_losses = None
        self.num_samples = None

    def get_main_loss_name(self):
        return self.__name__

    def on_training_epoch_start(self, state):
        """
        Initializes the array with batches of loss values
        :param state: the shared State object
        """
        self.batch_losses = []
        self.num_samples = 0

    def on_training_batch_end(self, state):
        """
        Updates the array of batch losses
        :param state: the shared State object
        """
        self.batch_losses.append(state.batch_loss[self.__name__].item() * state.batch_num_targets)
        self.num_samples += state.batch_num_targets

    def on_training_epoch_end(self, state):
        """
        Computes a loss value for the entire epoch
        :param state: the shared State object
        """
        state.update(epoch_loss={self.__name__: torch.tensor(self.batch_losses).sum()/self.num_samples})
        self.batch_losses = None
        self.num_samples = None

    def on_eval_epoch_start(self, state):
        """
        Initializes the array with batches of loss values
        :param state: the shared State object
        """
        self.batch_losses = []
        self.num_samples = 0

    def on_eval_epoch_end(self, state):
        """
        Computes a loss value for the entire epoch
        :param state: the shared State object
        """
        state.update(epoch_loss={self.__name__: torch.tensor(self.batch_losses).sum()/self.num_samples})
        self.batch_losses = None

    def on_eval_batch_end(self, state):
        """
        Updates the array of batch losses
        :param state: the shared State object
        """
        self.batch_losses.append(state.batch_loss[self.__name__].item() * state.batch_num_targets)
        self.num_samples += state.batch_num_targets

    def on_compute_metrics(self, state):
        """
        Computes the loss
        :param state: the shared State object
        """
        outputs, targets = state.batch_outputs, state.batch_targets
        loss_output = self.forward(targets, *outputs)

        if isinstance(loss_output, tuple):
            # Allow loss to produce intermediate results that speed up
            # Score computation. Loss callback MUST occur before the score one.
            loss, extra = loss_output
            state.update(batch_loss_extra={self.__name__: extra})
        else:
            loss = loss_output
        state.update(batch_loss={self.__name__: loss})

    def on_backward(self, state):
        """
        Computes the gradient of the computation graph
        :param state: the shared State object
        """
        try:
            state.batch_loss[self.__name__].backward()
        except Exception as e:
            # Here we catch potential multiprocessing related issues
            # see https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork
            print(e)

    def forward(self, targets, *outputs):
        """
        Computes the loss for a batch of output/target valies
        :param targets:
        :param outputs: a tuple of outputs returned by a model
        :return: loss and accuracy values
        """
        raise NotImplementedError('To be implemented by a subclass')


class AdditiveLoss(Loss):
    """
    MultiLoss combines an arbitrary number of Loss objects to perform backprop without having to istantiate a new class.
    The final loss is formally defined as the sum of the individual losses.
    """
    __name__ = "Additive Loss"
    op = operator.lt  # less than to determine improvement

    def _istantiate_loss(self, loss):
        if isinstance(loss, dict):
            args = loss["args"]
            return s2c(loss['class_name'])(**args)
        else:
            return s2c(loss)()

    def __init__(self, **losses):
        super().__init__()
        self.losses = [self._istantiate_loss(loss) for loss in losses.values()]

    def on_training_epoch_start(self, state):
        self.batch_losses = {l.__name__: [] for l in [self] + self.losses}
        self.num_targets = 0

    def on_training_batch_end(self, state):
        for k,v in state.batch_loss.items():
            self.batch_losses[k].append(v.item() * state.batch_num_targets)
        self.num_targets += state.batch_num_targets

    def on_training_epoch_end(self, state):
        state.update(epoch_loss={l.__name__: torch.tensor(self.batch_losses[l.__name__]).sum()/self.num_targets
                                  for l in [self] + self.losses})
        self.batch_losses = None
        self.num_targets = None

    def on_eval_epoch_start(self, state):
        self.batch_losses = {l.__name__: [] for l in [self] + self.losses}
        self.num_targets = 0

    def on_eval_epoch_end(self, state):
        state.update(epoch_loss={l.__name__: torch.tensor(self.batch_losses[l.__name__]).sum() / self.num_targets
                          for l in [self] + self.losses})
        self.batch_losses = None
        self.num_targets = None

    def on_eval_batch_end(self, state):
        for k,v in state.batch_loss.items():
            self.batch_losses[k].append(v.item() * state.batch_num_targets)
        self.num_targets += state.batch_num_targets

    def on_compute_metrics(self, state):
        """
        Computes the loss
        :param state: the shared State object
        """
        outputs, targets = state.batch_outputs, state.batch_targets
        loss = {}
        extra = {}
        loss_sum = 0.
        for l in self.losses:
            single_loss = l.forward(targets, *outputs)
            if isinstance(single_loss, tuple):
                # Allow loss to produce intermediate results that speed up
                # Score computation. Loss callback MUST occur before the score one.
                loss_output, loss_extra = single_loss
                extra[single_loss.__name__] = loss_extra
                state.update(batch_loss_extra=extra)
            else:
                loss_output = single_loss
            loss[l.__name__] = loss_output
            loss_sum += loss_output

        loss[self.__name__] = loss_sum
        state.update(batch_loss=loss)


class ClassificationLoss(Loss):
    __name__ = 'Classification Loss'

    def __init__(self):
        super().__init__()
        self.loss = None

    def forward(self, targets, *outputs):
        outputs = outputs[0]

        # print(outputs.shape, targets.shape)

        loss = self.loss(outputs, targets)
        return loss


class RegressionLoss(Loss):
    __name__ = 'Regression Loss'


    def __init__(self):
        super().__init__()
        self.loss = None

    def forward(self, targets, *outputs):
        outputs = outputs[0]
        loss = self.loss(outputs.squeeze(), targets.squeeze())
        return loss


class BinaryClassificationLoss(ClassificationLoss):
    __name__ = 'Binary Classification Loss'

    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction=reduction)


class MulticlassClassificationLoss(ClassificationLoss):
    __name__ = 'Multiclass Classification Loss'

    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction=reduction)


class MeanSquareErrorLoss(RegressionLoss):
    __name__ = 'MSE'

    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss = MSELoss(reduction=reduction)


class MeanAverageErrorLoss(RegressionLoss):
    __name__ = 'MAE'

    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss = L1Loss(reduction=reduction)

class Trento6d93_31_Loss(RegressionLoss):
    __name__ = 'Trento6d93_31_Loss'

    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss = L1Loss(reduction=reduction)

    def forward(self, targets, *outputs):
        outputs = torch.relu(outputs[0])
        loss = self.loss(torch.log(1+outputs), torch.log(1+targets))
        return loss


class CGMMLoss(Loss):
    __name__ = 'CGMM Loss'

    def __init__(self):
        super().__init__()
        self.old_likelihood = -float('inf')
        self.new_likelihood = None

    def on_training_batch_end(self, state):
        self.batch_losses.append(state.batch_loss[self.__name__].item())
        if state.model.is_graph_classification:
            self.num_samples += state.batch_num_targets
        else:
            # This works for unsupervised CGMM
            self.num_samples += state.batch_num_nodes

    def on_training_epoch_end(self, state):
        super().on_training_epoch_end(state)

        if (state.epoch_loss[self.__name__].item() - self.old_likelihood) < 0:
            pass
            #tate.stop_training = True
        self.old_likelihood = state.epoch_loss[self.__name__].item()

    def on_eval_batch_end(self, state):
        self.batch_losses.append(state.batch_loss[self.__name__].item())
        if state.model.is_graph_classification:
            self.num_samples += state.batch_num_targets
        else:
            # This works for unsupervised CGMM
            self.num_samples += state.batch_num_nodes

    # Simply ignore targets
    def forward(self, targets, *outputs):
        likelihood = outputs[2]
        return likelihood

    def on_backward(self, state):
        pass
    

class IOCGMMLoss(CGMMLoss):
    __name__ = 'IO-CGMM Loss'

    def on_backward(self, state):
        loss = -state.batch_loss[self.__name__]
        loss.backward()


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
