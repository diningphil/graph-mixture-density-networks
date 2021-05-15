import os
from pathlib import Path

import torch
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from training.engine import TrainingEngine
from training.event.handler import EventHandler


class Plotter(EventHandler):
    """
    Plotter is the main event handler for plotting at training time.
    """
    __name__ = 'plotter'

    def __init__(self, exp_path, **kwargs):
        super().__init__()
        self.exp_path = exp_path

        if not os.path.exists(Path(self.exp_path, 'tensorboard')):
            os.makedirs(Path(self.exp_path, 'tensorboard'))
        self.writer = SummaryWriter(log_dir=Path(self.exp_path, 'tensorboard'))

    def on_epoch_end(self, state):

        for k, v in state.epoch_results['losses'].items():
            loss_scalars = {}
            # Remove training/validation/test prefix (coupling with Engine)
            loss_name = ' '.join(k.split('_')[1:])
            if TrainingEngine.TRAINING in k:
                loss_scalars[f'{TrainingEngine.TRAINING}'] = v
            elif TrainingEngine.VALIDATION in k:
                loss_scalars[f'{TrainingEngine.VALIDATION}'] = v
            elif TrainingEngine.TEST in k:
                loss_scalars[f'{TrainingEngine.TEST}'] = v

            self.writer.add_scalars(loss_name, loss_scalars, state.epoch)


        for k, v in state.epoch_results['scores'].items():
            score_scalars = {}
            # Remove training/validation/test prefix (coupling with Engine)
            score_name = ' '.join(k.split('_')[1:])
            if TrainingEngine.TRAINING in k:
                score_scalars[f'{TrainingEngine.TRAINING}'] = v
            elif TrainingEngine.VALIDATION in k:
                score_scalars[f'{TrainingEngine.VALIDATION}'] = v
            elif TrainingEngine.TEST in k:
                score_scalars[f'{TrainingEngine.TEST}'] = v

            self.writer.add_scalars(score_name, score_scalars, state.epoch)


    def on_fit_end(self, state):
        self.writer.close()


class GMDNLinkPredictionPlotter(Plotter):

    def on_fit_end(self, state):
        # This will close the writers
        super().on_fit_end(state)

        train_embeddings = state.best_epoch_results[f'{TrainingEngine.TRAINING}_embeddings_tuple']
        #val_embeddings = state.best_epoch_results[f'{TrainingEngine.VALIDATION}_embeddings_tuple'] if f'{TrainingEngine.VALIDATION}_embeddings_tuple' in state.best_epoch_results else None
        #test_embeddings = state.best_epoch_results[f'{TrainingEngine.TEST}_embeddings_tuple'] if f'{TrainingEngine.TEST}_embeddings_tuple' in state.best_epoch_results else None

        # Assume single graph
        weights, mu, var, _, _, _ = train_embeddings[0].x
        np.save(Path(self.exp_path, "weights"), weights)
        np.save(Path(self.exp_path, "mu"), mu)
        np.save(Path(self.exp_path, "var"), var)


class IONeuralCGMMPlotter(Plotter):
    def __init__(self, exp_path, **kwargs):
        super().__init__(exp_path)
        self.targets = None
        self.pred = None

    def on_fit_end(self, state):
        # This will close the writers
        super().on_fit_end(state)

        training = 'training'
        validation = 'validation'
        test = 'test'
        depth = state.model.depth
        train_embeddings = state.best_epoch_results[f'{training}_embeddings_tuple']
        val_embeddings = state.best_epoch_results[f'{validation}_embeddings_tuple'] if f'{validation}_embeddings_tuple' in state.best_epoch_results else None
        test_embeddings = state.best_epoch_results[f'{test}_embeddings_tuple'] if f'{test}_embeddings_tuple' in state.best_epoch_results else None

        train_states = torch.cat(train_embeddings[3], dim=0).detach().cpu().numpy()
        train_preds = torch.cat(train_embeddings[0], dim=0).detach().cpu().numpy()

        if val_embeddings is not None:
            val_states = torch.cat(val_embeddings[3], dim=0).detach().cpu().numpy()
            val_preds = torch.cat(val_embeddings[0], dim=0).detach().cpu().numpy()
        else:
            val_states, val_preds = None, None

        if test_embeddings is not None:
            test_states = torch.cat(test_embeddings[3], dim=0).detach().cpu().numpy()
            test_preds = torch.cat(test_embeddings[0], dim=0).detach().cpu().numpy()
        else:
            test_states, test_preds = None, None

        for dataset, states, preds in [(training, train_states, train_preds), (validation, val_states, val_preds), (test, test_states, test_preds)]:

            if states is not None:
                plt.figure()
                np.save(Path(self.exp_path, f"{dataset}_embeddings_{depth}"), states)
                sns.heatmap(states)
                #plt.savefig(Path(self.exp_path, f"{dataset}_embeddings_{depth}.eps"))
                plt.savefig(Path(self.exp_path, f"{dataset}_embeddings_{depth}.png"))
                plt.close()

            if preds is not None:
                plt.figure()
                np.save(Path(self.exp_path, f"{dataset}_preds_{depth}"), preds)
                sns.heatmap(preds)
                #plt.savefig(Path(self.exp_path, f"{dataset}_preds_{depth}.eps"))
                plt.savefig(Path(self.exp_path, f"{dataset}_preds_{depth}.png"))
                plt.close()
