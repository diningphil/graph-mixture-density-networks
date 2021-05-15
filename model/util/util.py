import math
import torch
from torch_scatter import scatter

pi = torch.FloatTensor([math.pi])


def _make_one_hot(labels, C):
    device = labels.get_device()
    device = 'cpu' if device == -1 else device
    one_hot = torch.zeros(labels.size(0), C).to(device)
    one_hot[torch.arange(labels.size(0)).to(device), labels] = 1
    return one_hot


def global_min_pool(x, batch, size=None):
    r"""Returns batch-wise graph-level-outputs by taking the channel-wise
    maximum across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \mathrm{max}_{n=1}^{N_i} \, \mathbf{x}_n

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    size = batch.max().item() + 1 if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='min')


def multivariate_diagonal_pdf(data, mean, var):
    '''
    Multivariate case, DIAGONAL cov. matrix. Computes probability distribution for each data point
    :param data: a vector of values
    :param mean: vector of mean values, size F
    :param var:  vector of variance values, size F
    :return:
    '''
    tmp =  torch.log(2 * pi)
    try:
        device = 'cuda:' + str(data.get_device())
        tmp = tmp.to(device)
    except Exception as e:
        pass

    diff = (data.float() - mean)
    log_normaliser = -0.5 * (tmp + torch.log(var))
    log_num = - (diff * diff) / (2 * var)
    log_probs = torch.sum(log_num + log_normaliser, dim=1)
    probs = torch.exp(log_probs)
    return probs
