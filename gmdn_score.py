from typing import List

import torch
from pydgn.training.callback.metric import Metric


class DirichletPriorScore(Metric):


    @property
    def name(self) -> str:
        return 'Dirichlet Prior'

    def __init__(self):
        super().__init__()

    def forward(self, targets: torch.Tensor, *outputs: List[torch.Tensor], batch_loss_extra: dict = None) -> dict:
        return outputs[4]


class LikelihoodScore(Metric):
    @property
    def name(self) -> str:
        return 'True Log Likelihood'

    def __init__(self):
        super().__init__()

    def forward(self, targets: torch.Tensor, *outputs: List[torch.Tensor], batch_loss_extra: dict = None) -> dict:
        return outputs[3]
