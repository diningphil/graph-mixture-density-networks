import numpy as np
from pydgn.data.splitter import Splitter, InnerFold, OuterFold
from sklearn.model_selection import train_test_split


class SIRSplitter(Splitter):

    def __init__(self, n_outer_folds=1, n_inner_folds=1, seed=42, **kwargs):
        """
        Initializes the splitter
        :param n_outer_folds: number of outer folds (risk assessment). Must be 1 (holdout)
        :param n_inner_folds: number of inner folds (model selection). Must be 1 (holdout)
        :param seed: random seed for reproducibility
        """

        self.stratify = False
        self.shuffle = False
        assert n_outer_folds == 1, "Only one outer fold is allowed"
        assert n_inner_folds == 1, "Only one inner fold is allowed"

        super().__init__(n_outer_folds, n_inner_folds, seed, self.stratify, self.shuffle)

    def split(self, dataset, targets=None, test_ratio=0.1):
        """
        Computes the splits
        :param dataset: the Dataset object
        :param test_ratio: percentage of validation/test set when using an internal/external hold-out split. Default value is 0.1.
        :return:
        """
        idxs = range(len(dataset))

        if not self.processed:

            outer_idxs = np.array(idxs)

            train_idxs, test_idxs = train_test_split(outer_idxs, test_ratio=test_ratio, shuffle=False)

            inner_fold_splits = []

            inner_train_idxs, inner_val_idxs = train_test_split(train_idxs, test_ratio=test_ratio, shuffle=False)
            np.random.shuffle(inner_train_idxs)

            inner_fold = InnerFold(inner_train_idxs.tolist(), inner_val_idxs.tolist())
            inner_fold_splits.append(inner_fold)
            self.inner_folds.append(inner_fold_splits)

            np.random.shuffle(train_idxs)
            outer_fold = OuterFold(train_idxs.tolist(), test_idxs.tolist())
            self.outer_folds.append(outer_fold)

            self.processed = True
