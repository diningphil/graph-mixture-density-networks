from typing import List

import torch
from pydgn.training.callback.metric import Metric


class DirichletPriorScore(Metric):

    @property
    def name(self) -> str:
        return 'Dirichlet Prior'

    def forward(self, targets: torch.Tensor, *outputs: List[torch.Tensor], batch_loss_extra: dict = None) -> dict:
        return outputs[2][2]


class LogLikelihoodScore(Metric):
    @property
    def name(self) -> str:
        return 'Log Likelihood'

    def forward(self, targets: torch.Tensor, *outputs: List[torch.Tensor], batch_loss_extra: dict = None) -> dict:
        return outputs[2][1]


class DirichletPriorScore(Metric):

    @property
    def name(self) -> str:
        return 'Dirichlet Prior'

    def forward(self, targets: torch.Tensor, *outputs: List[torch.Tensor], batch_loss_extra: dict = None) -> dict:
        return outputs[2][2]


class MeanAverageError(Metric):
    @property
    def name(self) -> str:
        return 'MAE'

    def forward(self, targets: torch.Tensor, *outputs: List[torch.Tensor], batch_loss_extra: dict = None) -> dict:
        return torch.nn.functional.l1_loss(outputs[0].squeeze(dim=1), targets.squeeze(dim=1))
