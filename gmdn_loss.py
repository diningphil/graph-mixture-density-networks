from pydgn.training.callback.metric import Metric


class GMDNLoss(Metric):

    @property
    def name(self) -> str:
        return 'GMDN Loss'

    # Simply ignore targets
    def forward(self, targets, *outputs):
        likelihood = outputs[2][0]
        return likelihood

    def on_backward(self, state):
        loss = -state.batch_loss[self.name]
        loss.backward()