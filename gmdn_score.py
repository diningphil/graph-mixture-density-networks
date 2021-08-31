import torch
from pydgn.training.callback.score import Score

from sklearn.metrics import roc_auc_score, average_precision_score
from gmdn_loss import L2GMMLinkPredictionLoss, JeffreyGMMLinkPredictionLoss, BhattacharyyaGMMLinkPredictionLoss


class DirichletPriorScore(Score):
    __name__ ='Dirichlet Prior'

    def __init__(self):
        super().__init__()

    def _score_fun(self, targets, *outputs, batch_loss_extra):
        return outputs[4]


class LikelihoodScore(Score):
    __name__ ='True Log Likelihood'

    def __init__(self):
        super().__init__()

    def _score_fun(self, targets, *outputs, batch_loss_extra):
        return outputs[3]


class DotProductLinkPredictionROCAUCScore(Score):

    __name__ = "DotProduct Link Prediction ROC AUC Score"

    def __init__(self):
        super().__init__()

    def _score_fun(self, targets, *outputs, batch_loss_extra):
        node_embs = outputs[1]
        _, pos_edges, neg_edges = targets[0]

        loss_edge_index = torch.cat((pos_edges, neg_edges), dim=1)
        loss_target = torch.cat((torch.ones(pos_edges.shape[1]),
                                  torch.zeros(neg_edges.shape[1])))

        # Taken from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/link_pred.py
        x_j = torch.index_select(node_embs, 0, loss_edge_index[0])
        x_i = torch.index_select(node_embs, 0, loss_edge_index[1])
        link_logits = torch.einsum("ef,ef->e", x_i, x_j)

        link_probs = torch.sigmoid(link_logits)
        link_probs = link_probs.detach().cpu().numpy()
        loss_target = loss_target.detach().cpu().numpy()
        score = torch.tensor([roc_auc_score(loss_target, link_probs)])
        return score


class DotProductLinkPredictionAPScore(Score):

    __name__ = "DotProduct Link Prediction AP Score"

    def __init__(self):
        super().__init__()

    def _score_fun(self, targets, *outputs, batch_loss_extra):
        node_embs = outputs[1]
        _, pos_edges, neg_edges = targets[0]

        loss_edge_index = torch.cat((pos_edges, neg_edges), dim=1)
        loss_target = torch.cat((torch.ones(pos_edges.shape[1]),
                                  torch.zeros(neg_edges.shape[1])))

        # Taken from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/link_pred.py
        x_j = torch.index_select(node_embs, 0, loss_edge_index[0])
        x_i = torch.index_select(node_embs, 0, loss_edge_index[1])
        link_logits = torch.einsum("ef,ef->e", x_i, x_j)

        link_probs = torch.sigmoid(link_logits)
        link_probs = link_probs.detach().cpu().numpy()
        loss_target = loss_target.detach().cpu().numpy()
        score = torch.tensor([average_precision_score(loss_target, link_probs)])
        return score


class L2GMMLinkPredictionROCAUCScore(Score):
    __name__ ='L2 GMM Link Prediction ROC AUC'

    def __init__(self):
        super().__init__()
        self.threshold = 0.5

    def _score_fun(self, targets, *outputs, batch_loss_extra):
        """
        This works for mixtures of 1-D Gaussians
        """
        weights, mu, var = outputs[1][0], outputs[1][1], outputs[1][2]
        node_embs = outputs[1][3]
        _, pos_edges, neg_edges = targets[0]

        loss_edge_index = torch.cat((pos_edges, neg_edges), dim=1)
        loss_target = torch.cat((torch.ones(pos_edges.shape[1]),
                                  torch.zeros(neg_edges.shape[1])))

        squared_distance = L2GMMLinkPredictionLoss.compute_loss(mu, var, weights, loss_edge_index)
        distance = torch.sqrt(squared_distance)

        #prob = (distance < self.threshold).float()
        prob = 1./(1. + distance)

        return torch.tensor([roc_auc_score(loss_target.detach().cpu().numpy(), prob.detach().cpu().numpy())])


class L2GMMLinkPredictionAPScore(Score):
    __name__ ='L2 GMM Link Prediction AP'

    def __init__(self):
        super().__init__()
        self.threshold = 0.5

    def _score_fun(self, targets, *outputs, batch_loss_extra):
        """
        This works for mixtures of 1-D Gaussians
        """
        weights, mu, var = outputs[1][0], outputs[1][1], outputs[1][2]
        node_embs = outputs[1][3]
        _, pos_edges, neg_edges = targets[0]

        loss_edge_index = torch.cat((pos_edges, neg_edges), dim=1)
        loss_target = torch.cat((torch.ones(pos_edges.shape[1]),
                                  torch.zeros(neg_edges.shape[1])))

        squared_distance = L2GMMLinkPredictionLoss.compute_loss(mu, var, weights, loss_edge_index)
        distance = torch.sqrt(squared_distance)

        #prob = (distance < self.threshold).float()
        prob = 1./(1. + distance)

        return torch.tensor([average_precision_score(loss_target.detach().cpu().numpy(), prob.detach().cpu().numpy())])


class JeffreyGMMLinkPredictionROCAUCScore(Score):
    __name__ ='Jeffrey GMM Link Prediction ROC AUC'

    def __init__(self):
        super().__init__()
        self.threshold = 0.5

    def _score_fun(self, targets, *outputs, batch_loss_extra):
        """
        This works for mixtures of 1-D Gaussians
        """
        weights, mu, var = outputs[1][0], outputs[1][1], outputs[1][2]
        node_embs = outputs[1][3]
        _, pos_edges, neg_edges = targets[0]

        loss_edge_index = torch.cat((pos_edges, neg_edges), dim=1)
        loss_target = torch.cat((torch.ones(pos_edges.shape[1]),
                                  torch.zeros(neg_edges.shape[1])))

        distance = JeffreyGMMLinkPredictionLoss.compute_loss(mu, var, weights, loss_edge_index)
        prob = 1./(1. + distance)

        return torch.tensor([roc_auc_score(loss_target.detach().cpu().numpy(), prob.detach().cpu().numpy())])


class JeffreyGMMLinkPredictionAPScore(Score):
    __name__ ='Jeffrey GMM Link Prediction AP'

    def __init__(self):
        super().__init__()
        self.threshold = 0.5

    def _score_fun(self, targets, *outputs, batch_loss_extra):
        """
        This works for mixtures of 1-D Gaussians
        """
        weights, mu, var = outputs[1][0], outputs[1][1], outputs[1][2]
        node_embs = outputs[1][3]
        _, pos_edges, neg_edges = targets[0]

        loss_edge_index = torch.cat((pos_edges, neg_edges), dim=1)
        loss_target = torch.cat((torch.ones(pos_edges.shape[1]),
                                  torch.zeros(neg_edges.shape[1])))

        distance = JeffreyGMMLinkPredictionLoss.compute_loss(mu, var, weights, loss_edge_index)
        prob = 1./(1. + distance)

        return torch.tensor([average_precision_score(loss_target.detach().cpu().numpy(), prob.detach().cpu().numpy())])


class BhattacharyyaGMMLinkPredictionROCAUCScore(Score):
    __name__ ='Bhattacharyya GMM Link Prediction ROC AUC'

    def __init__(self):
        super().__init__()
        self.threshold = 0.5

    def _score_fun(self, targets, *outputs, batch_loss_extra):
        """
        This works for mixtures of 1-D Gaussians
        """
        weights, mu, var = outputs[1][0], outputs[1][1], outputs[1][2]
        node_embs = outputs[1][3]
        _, pos_edges, neg_edges = targets[0]

        loss_edge_index = torch.cat((pos_edges, neg_edges), dim=1)
        loss_target = torch.cat((torch.ones(pos_edges.shape[1]),
                                  torch.zeros(neg_edges.shape[1])))

        distance = BhattacharyyaGMMLinkPredictionLoss.compute_loss(mu, var, weights, loss_edge_index)
        prob = 1./(1. + distance)

        return torch.tensor([roc_auc_score(loss_target.detach().cpu().numpy(), prob.detach().cpu().numpy())])


class BhattacharyyaGMMLinkPredictionAPScore(Score):
    __name__ ='Bhattacharyya GMM Link Prediction AP'

    def __init__(self):
        super().__init__()
        self.threshold = 0.5

    def _score_fun(self, targets, *outputs, batch_loss_extra):
        """
        This works for mixtures of 1-D Gaussians
        """
        weights, mu, var = outputs[1][0], outputs[1][1], outputs[1][2]
        node_embs = outputs[1][3]
        _, pos_edges, neg_edges = targets[0]

        loss_edge_index = torch.cat((pos_edges, neg_edges), dim=1)
        loss_target = torch.cat((torch.ones(pos_edges.shape[1]),
                                  torch.zeros(neg_edges.shape[1])))

        distance = BhattacharyyaGMMLinkPredictionLoss.compute_loss(mu, var, weights, loss_edge_index)
        prob = 1./(1. + distance)

        return torch.tensor([average_precision_score(loss_target.detach().cpu().numpy(), prob.detach().cpu().numpy())])


class DotProductGMMLinkPredictionROCAUCScore(Score):
    __name__ ='DotProduct GMM Link Prediction ROC AUC'

    def __init__(self):
        super().__init__()
        self.threshold = 0.5

    def _score_fun(self, targets, *outputs, batch_loss_extra):
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

        link_probs = torch.sigmoid(link_logits)
        link_probs = link_probs.detach().cpu().numpy()
        loss_target = loss_target.detach().cpu().numpy()
        score = torch.tensor([roc_auc_score(loss_target, link_probs)])
        return score


class DotProductGMMLinkPredictionAPScore(Score):
    __name__ ='DotProduct GMM Link Prediction AP'

    def __init__(self):
        super().__init__()
        self.threshold = 0.5

    def _score_fun(self, targets, *outputs, batch_loss_extra):
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

        link_probs = torch.sigmoid(link_logits)
        link_probs = link_probs.detach().cpu().numpy()
        loss_target = loss_target.detach().cpu().numpy()
        score = torch.tensor([average_precision_score(loss_target, link_probs)])
        return score
